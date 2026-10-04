"""Neural multi-task model over the static (drug-free) modalities.

Encoder: each modality's sparse vector is projected to ``d`` dimensions
(a learned embedding of HPO terms, genes, pathways, ontology nodes, words),
and an attention layer pools the modalities a disease actually has into one
disease embedding ``z``.  Modality dropout makes the pooled embedding robust
to the missing annotations typical of rare diseases (and of new diseases).

Pair head: an MLP over the symmetric features ``[z_a * z_b, |z_a - z_b|]``
plus the raw per-modality cosine similarities ``S`` and availability ``A``,
so it can always fall back on the transparent cosine evidence.

Tasks (one logit each):
  therapeutic  pairs that already share an orphan-designated drug (before the
               training date) -- the target task;
  sibling      cohort diseases with the same direct Orphanet parent
               (ontology and name hidden, they encode the classification);
  gene         cohort diseases sharing a causal gene (gene and pathway hidden).
The auxiliary tasks teach the encoder general disease relatedness from far
more pairs than the regulatory history alone provides.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from features import StaticSimilarity
from modalities import MODALITIES
from world import Snapshot, World, gene_pairs, sibling_pairs

TASKS = ("therapeutic", "sibling", "gene")
TASK_MASKS = {"therapeutic": (), "sibling": ("ontology", "name"), "gene": ("gene", "pathway")}

DEFAULT_CONFIG: dict[str, Any] = {
    "dim": 64,
    "z_dim": 128,
    "hidden": 256,
    "dropout": 0.2,
    "modality_dropout": 0.2,
    "lr": 2e-3,
    "weight_decay": 1e-4,
    "batch": 1024,
    "epochs": 30,
    "patience": 5,
    "aux_weight": 0.5,
    "negatives_per_side": 20,
    "aux_pairs": 20_000,
    "aux_negatives": 3,
    "holdout": 0.1,
    "seeds": 3,
}


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_torch(matrix: sp.csr_matrix, device: torch.device) -> torch.Tensor:
    coo = matrix.tocoo()
    indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    values = torch.from_numpy(coo.data.astype(np.float32))
    return torch.sparse_coo_tensor(indices, values, coo.shape, device=device, check_invariants=False).coalesce()


class StaticNet(nn.Module):
    def __init__(self, dims: dict[str, int], config: dict[str, Any]):
        super().__init__()
        d, dz, hidden, p = config["dim"], config["z_dim"], config["hidden"], config["dropout"]
        self.modalities = tuple(dims)
        self.projection = nn.ParameterDict(
            {m: nn.Parameter(torch.randn(v, d) * (1.0 / math.sqrt(max(v, 1)) + 0.01)) for m, v in dims.items()}
        )
        self.bias = nn.ParameterDict({m: nn.Parameter(torch.zeros(d)) for m in dims})
        self.norm = nn.ModuleDict({m: nn.LayerNorm(d) for m in dims})
        self.modality_embedding = nn.Parameter(torch.randn(len(dims), d) * 0.02)
        self.attention = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1))
        self.null = nn.Parameter(torch.zeros(d))
        self.output = nn.Sequential(nn.Linear(d, dz), nn.GELU(), nn.Linear(dz, dz))
        self.output_norm = nn.LayerNorm(dz)
        m = len(dims)
        self.head = nn.Sequential(
            nn.Linear(2 * dz + 2 * m, hidden),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(p),
        )
        self.task_output = nn.Linear(hidden // 2, len(TASKS))

    def encode(
        self,
        inputs: dict[str, torch.Tensor],
        available: torch.Tensor,
        masked: torch.Tensor,
        modality_dropout: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Disease embeddings (N x z_dim) and attention weights (N x modalities)."""

        hidden = torch.stack(
            [self.norm[m](F.gelu(torch.sparse.mm(inputs[m], self.projection[m]) + self.bias[m])) for m in self.modalities],
            dim=1,
        )
        scores = self.attention(hidden + self.modality_embedding).squeeze(-1)
        use = available & ~masked[None, :]
        if modality_dropout > 0 and self.training:
            use = use & (torch.rand_like(scores) >= modality_dropout)
        scores = scores.masked_fill(~use, float("-inf"))
        scores = torch.cat([scores, torch.zeros(scores.shape[0], 1, device=scores.device)], dim=1)
        weights = torch.softmax(scores, dim=1)
        pooled = (weights[:, :-1, None] * hidden).sum(dim=1) + weights[:, -1:] * self.null
        return self.output_norm(self.output(pooled)), weights[:, :-1]

    def pair(self, za: torch.Tensor, zb: torch.Tensor, S: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        x = torch.cat([za * zb, (za - zb).abs(), S, A], dim=-1)
        return self.task_output(self.head(x))


@dataclass
class PairPool:
    a: np.ndarray
    b: np.ndarray
    y: np.ndarray
    S: np.ndarray
    A: np.ndarray


def _negatives(
    anchors: np.ndarray,
    count: int,
    pools: list[np.ndarray],
    excluded: sp.csr_matrix,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """``count`` negatives per anchor, alternating between candidate pools."""

    a = np.repeat(anchors, count)
    b = np.empty_like(a)
    for k, pool in enumerate(pools):
        slots = np.arange(k, a.size, len(pools))
        b[slots] = pool[rng.integers(0, pool.size, slots.size)]
    bad = np.asarray(excluded[a, b]).ravel() > 0
    for _ in range(10):
        if not bad.any():
            break
        b[bad] = pools[0][rng.integers(0, pools[0].size, int(bad.sum()))]
        bad = np.asarray(excluded[a, b]).ravel() > 0
    keep = ~bad
    return a[keep], b[keep]


def _pool(sim: StaticSimilarity, a: np.ndarray, b: np.ndarray, y: np.ndarray, masked: tuple[str, ...]) -> PairPool:
    S, A = sim.pairs(a, b, masked)
    return PairPool(a=a, b=b, y=y.astype(np.float32), S=S, A=A)


def build_pools(
    world: World, sim: StaticSimilarity, snapshot: Snapshot, config: dict[str, Any], rng: np.random.Generator
) -> dict[str, PairPool]:
    n = world.n
    excluded = ((snapshot.adjacency + world.nested + sp.identity(n, format="csr")) > 0).tocsr()
    everyone = np.arange(n)
    warm = np.flatnonzero(snapshot.warm)
    pools: dict[str, PairPool] = {}

    rel = snapshot.relations
    pa = np.array([world.index[x] for x in rel["a"]], dtype=np.int64)
    pb = np.array([world.index[x] for x in rel["b"]], dtype=np.int64)
    k = config["negatives_per_side"]
    candidate_pools = [everyone, warm] if warm.size else [everyone]
    na1, nb1 = _negatives(pa, k, candidate_pools, excluded, rng)
    na2, nb2 = _negatives(pb, k, candidate_pools, excluded, rng)
    a = np.r_[pa, na1, na2]
    b = np.r_[pb, nb1, nb2]
    y = np.r_[np.ones(pa.size), np.zeros(na1.size + na2.size)]
    pools["therapeutic"] = _pool(sim, a, b, y, TASK_MASKS["therapeutic"])

    cohort = np.flatnonzero(~world.is_group)
    aux_excluded = ((world.nested + sp.identity(n, format="csr")) > 0).tocsr()
    for task, builder in (("sibling", sibling_pairs), ("gene", gene_pairs)):
        positives = builder(world, config["aux_pairs"], rng)
        na, nb = _negatives(positives[:, 0], config["aux_negatives"], [cohort], aux_excluded, rng)
        a = np.r_[positives[:, 0], na]
        b = np.r_[positives[:, 1], nb]
        y = np.r_[np.ones(len(positives)), np.zeros(na.size)]
        pools[task] = _pool(sim, a, b, y, TASK_MASKS[task])
    return pools


@dataclass
class StaticModel:
    """One trained network plus cached embeddings of every node."""

    state: dict[str, Any]
    dims: dict[str, int]
    config: dict[str, Any]
    z: np.ndarray
    attention: np.ndarray
    history: list[dict[str, float]] = field(default_factory=list)
    _net: Optional[StaticNet] = field(default=None, repr=False)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_net"] = None
        return state

    @property
    def net(self) -> StaticNet:
        if self._net is None:
            net = StaticNet(self.dims, self.config)
            net.load_state_dict(self.state)
            self._net = net.to(_device()).eval()
        return self._net

    @torch.no_grad()
    def pair_logits(self, z_a: np.ndarray, z_b: np.ndarray, S: np.ndarray, A: np.ndarray, task: int = 0) -> np.ndarray:
        device = _device()
        out = np.empty(len(z_a), dtype=np.float32)
        step = 262_144
        for start in range(0, len(z_a), step):
            sl = slice(start, start + step)
            logits = self.net.pair(
                torch.from_numpy(z_a[sl]).to(device),
                torch.from_numpy(z_b[sl]).to(device),
                torch.from_numpy(S[sl]).to(device),
                torch.from_numpy(A[sl].astype(np.float32)).to(device),
            )
            out[sl] = logits[:, task].float().cpu().numpy()
        return out

    def block(self, rows: np.ndarray, S: np.ndarray, A: np.ndarray, z_rows: Optional[np.ndarray] = None) -> np.ndarray:
        """Therapeutic logits of ``rows`` against every node; S, A are (rows, N, M)."""

        nq, n, m = S.shape
        z_q = self.z[rows] if z_rows is None else z_rows
        z_a = np.repeat(z_q, n, axis=0)
        z_b = np.tile(self.z, (nq, 1))
        return self.pair_logits(z_a, z_b, S.reshape(-1, m), A.reshape(-1, m)).reshape(nq, n)


@torch.no_grad()
def _evaluate(net, inputs, available, pool: PairPool, device) -> float:
    from sklearn.metrics import roc_auc_score

    net.eval()
    z, _ = net.encode(inputs, available, torch.zeros(len(MODALITIES), dtype=torch.bool, device=device))
    logits = net.pair(
        z[torch.from_numpy(pool.a).to(device)],
        z[torch.from_numpy(pool.b).to(device)],
        torch.from_numpy(pool.S).to(device),
        torch.from_numpy(pool.A.astype(np.float32)).to(device),
    )[:, 0]
    net.train()
    return float(roc_auc_score(pool.y, logits.cpu().numpy()))


def _split(pool: PairPool, fraction: float, rng: np.random.Generator) -> tuple[PairPool, PairPool]:
    """Hold out whole anchor diseases so the early-stopping score is not memorized."""

    anchors = np.unique(pool.a[pool.y == 1])
    held = set(rng.choice(anchors, max(1, int(round(fraction * anchors.size))), replace=False).tolist())
    mask = np.array([x in held for x in pool.a])

    def take(keep: np.ndarray) -> PairPool:
        return PairPool(pool.a[keep], pool.b[keep], pool.y[keep], pool.S[keep], pool.A[keep])

    return take(~mask), take(mask)


def train_static(
    world: World,
    sim: StaticSimilarity,
    pools: dict[str, PairPool],
    seed: int,
    config: Optional[dict[str, Any]] = None,
    verbose: bool = False,
) -> StaticModel:
    config = {**DEFAULT_CONFIG, **(config or {})}
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = _device()
    inputs = {m: _to_torch(world.matrices[m], device) for m in MODALITIES}
    available = torch.from_numpy(world.available).to(device)
    dims = {m: world.matrices[m].shape[1] for m in MODALITIES}
    net = StaticNet(dims, config).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    masks = {
        t: torch.tensor([m in TASK_MASKS[t] for m in MODALITIES], dtype=torch.bool, device=device) for t in TASKS
    }
    train_pool, holdout = _split(pools["therapeutic"], config["holdout"], rng)
    tensors = {}
    for task, pool in (("therapeutic", train_pool), ("sibling", pools["sibling"]), ("gene", pools["gene"])):
        tensors[task] = {
            "a": torch.from_numpy(pool.a).to(device),
            "b": torch.from_numpy(pool.b).to(device),
            "y": torch.from_numpy(pool.y).to(device),
            "S": torch.from_numpy(pool.S).to(device),
            "A": torch.from_numpy(pool.A.astype(np.float32)).to(device),
        }
    n_train = len(train_pool.y)
    batch = config["batch"]
    steps_per_epoch = max(1, math.ceil(n_train / batch))
    positive_rate = {t: float(tensors[t]["y"].mean()) for t in TASKS}
    best, best_state, history, bad_epochs = -1.0, None, [], 0
    for epoch in range(config["epochs"]):
        net.train()
        order = torch.randperm(n_train, device=device)
        total = 0.0
        for step in range(steps_per_epoch):
            loss = torch.zeros((), device=device)
            for t_index, task in enumerate(TASKS):
                data = tensors[task]
                if task == "therapeutic":
                    idx = order[step * batch : (step + 1) * batch]
                else:
                    idx = torch.randint(0, data["y"].shape[0], (batch,), device=device)
                z, _ = net.encode(inputs, available, masks[task], config["modality_dropout"])
                S, A = data["S"][idx], data["A"][idx]
                drop = torch.rand_like(S) < config["modality_dropout"] * 0.5
                S, A = S.masked_fill(drop, 0.0), A.masked_fill(drop, 0.0)
                logits = net.pair(z[data["a"][idx]], z[data["b"][idx]], S, A)[:, t_index]
                rate = positive_rate[task]
                weight = torch.where(data["y"][idx] > 0, 0.5 / rate, 0.5 / (1 - rate))
                task_loss = F.binary_cross_entropy_with_logits(logits, data["y"][idx], weight=weight)
                loss = loss + (1.0 if task == "therapeutic" else config["aux_weight"]) * task_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        auc = _evaluate(net, inputs, available, holdout, device)
        history.append({"epoch": epoch + 1, "loss": total / steps_per_epoch, "holdout_auc": auc})
        if verbose:
            print(f"    epoch {epoch + 1:2d} loss {total / steps_per_epoch:.4f} holdout AUC {auc:.4f}")
        if auc > best + 1e-4:
            best, bad_epochs = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= config["patience"]:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        z, attention = net.encode(inputs, available, torch.zeros(len(MODALITIES), dtype=torch.bool, device=device))
    model = StaticModel(
        state=best_state,
        dims=dims,
        config=config,
        z=z.cpu().numpy(),
        attention=attention.cpu().numpy(),
        history=history,
    )
    model._net = net
    return model


@dataclass
class StaticEnsemble:
    models: list[StaticModel]

    def block(self, rows: np.ndarray, S: np.ndarray, A: np.ndarray) -> np.ndarray:
        return np.mean([m.block(rows, S, A) for m in self.models], axis=0)

    def encode_new(self, world_matrices: dict[str, sp.csr_matrix], available: np.ndarray) -> list[np.ndarray]:
        """Embeddings of new (user-supplied) diseases under each member network."""

        device = _device()
        inputs = {m: _to_torch(world_matrices[m], device) for m in MODALITIES}
        avail = torch.from_numpy(available).to(device)
        out = []
        with torch.no_grad():
            for model in self.models:
                z, _ = model.net.encode(inputs, avail, torch.zeros(len(MODALITIES), dtype=torch.bool, device=device))
                out.append(z.cpu().numpy())
        return out

    def block_new(self, z_new: list[np.ndarray], S: np.ndarray, A: np.ndarray) -> np.ndarray:
        rows = np.zeros(S.shape[0], dtype=np.int64)
        return np.mean([m.block(rows, S, A, z_rows=z) for m, z in zip(self.models, z_new)], axis=0)


def train_ensemble(
    world: World, sim: StaticSimilarity, snapshot: Snapshot, config: Optional[dict[str, Any]] = None, verbose: bool = False
) -> StaticEnsemble:
    config = {**DEFAULT_CONFIG, **(config or {})}
    models = []
    for seed in range(config["seeds"]):
        rng = np.random.default_rng(1000 + seed)
        pools = build_pools(world, sim, snapshot, config, rng)
        models.append(train_static(world, sim, pools, seed, config, verbose))
    return StaticEnsemble(models)
