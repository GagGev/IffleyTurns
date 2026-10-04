"""The single v4 architecture: a multimodal disease embedding scored by cosine."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import joblib
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from common import MODEL_PATH
from data import Bundle
from modalities import MODALITIES, TASK_MASKS, DiseaseEncoder

torch.sparse.check_sparse_tensor_invariants.disable()

DEFAULT_CONFIG: dict[str, Any] = {
    "projection_dim": 64,
    "embedding_dim": 128,
    "dropout": 0.15,
    "modality_dropout": 0.20,
    "learning_rate": 2e-3,
    "weight_decay": 1e-4,
    "epochs": 80,
    "patience": 10,
    "margin": 0.15,
    "temperature": 0.12,
    "negatives": 4,
    "max_pairs_per_task": 20_000,
    "validation_negatives": 20,
    "seed": 0,
}
MAX_SIBLING_GROUP = 40
MAX_DISEASES_PER_GENE = 20


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_torch(matrix: sp.csr_matrix, target: torch.device) -> torch.Tensor:
    coo = matrix.tocoo()
    indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    values = torch.from_numpy(coo.data.astype(np.float32))
    return torch.sparse_coo_tensor(indices, values, coo.shape, device=target, check_invariants=False).coalesce()


class EmbeddingNet(nn.Module):
    """Project each sparse modality, attend over available modalities, normalize."""

    def __init__(self, dimensions: dict[str, int], config: dict[str, Any]):
        super().__init__()
        projection_dim = int(config["projection_dim"])
        embedding_dim = int(config["embedding_dim"])
        self.modalities = tuple(dimensions)
        self.projection = nn.ParameterDict(
            {
                name: nn.Parameter(
                    torch.randn(width, projection_dim) * (1 / math.sqrt(max(width, 1)) + 0.01)
                )
                for name, width in dimensions.items()
            }
        )
        self.bias = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(projection_dim)) for name in dimensions}
        )
        self.norm = nn.ModuleDict({name: nn.LayerNorm(projection_dim) for name in dimensions})
        self.modality_embedding = nn.Parameter(torch.randn(len(dimensions), projection_dim) * 0.02)
        self.attention = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.Tanh(),
            nn.Linear(projection_dim, 1),
        )
        self.null = nn.Parameter(torch.zeros(projection_dim))
        self.output = nn.Sequential(
            nn.Linear(projection_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(float(config["dropout"])),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.output_norm = nn.LayerNorm(embedding_dim)

    def encode(
        self,
        inputs: dict[str, torch.Tensor],
        available: torch.Tensor,
        masked: torch.Tensor,
        modality_dropout: float = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.stack(
            [
                self.norm[name](F.gelu(torch.sparse.mm(inputs[name], self.projection[name]) + self.bias[name]))
                for name in self.modalities
            ],
            dim=1,
        )
        scores = self.attention(hidden + self.modality_embedding).squeeze(-1)
        use = available & ~masked[None, :]
        if self.training and modality_dropout > 0:
            use = use & (torch.rand_like(scores) >= modality_dropout)
        scores = scores.masked_fill(~use, float("-inf"))
        scores = torch.cat([scores, torch.zeros(scores.shape[0], 1, device=scores.device)], dim=1)
        weights = torch.softmax(scores, dim=1)
        pooled = (weights[:, :-1, None] * hidden).sum(1) + weights[:, -1:] * self.null
        embedding = F.normalize(self.output_norm(self.output(pooled)), dim=-1)
        return embedding, weights[:, :-1]


@dataclass(frozen=True)
class RelationPairs:
    sibling: np.ndarray
    gene: np.ndarray

    def by_task(self) -> dict[str, np.ndarray]:
        return {"sibling": self.sibling, "gene": self.gene}


def build_relation_pairs(bundle: Bundle, ids: Sequence[str]) -> RelationPairs:
    index = {disease: i for i, disease in enumerate(ids)}
    children: dict[str, list[int]] = defaultdict(list)
    for parent, child in bundle.orphanet_edges:
        if child in index:
            children[parent].append(index[child])
    sibling: set[tuple[int, int]] = set()
    for members in children.values():
        members = sorted(set(members))
        if 2 <= len(members) <= MAX_SIBLING_GROUP:
            sibling.update(
                (members[a], members[b])
                for a in range(len(members))
                for b in range(a + 1, len(members))
            )

    by_gene: dict[str, list[int]] = defaultdict(list)
    for disease, genes in bundle.causal_genes.items():
        if disease in index:
            for gene in genes:
                by_gene[gene].append(index[disease])
    gene_pairs: set[tuple[int, int]] = set()
    for members in by_gene.values():
        members = sorted(set(members))
        if 2 <= len(members) <= MAX_DISEASES_PER_GENE:
            gene_pairs.update(
                (members[a], members[b])
                for a in range(len(members))
                for b in range(a + 1, len(members))
            )

    def array(values: set[tuple[int, int]]) -> np.ndarray:
        return np.asarray(sorted(values), dtype=np.int64).reshape(-1, 2)

    return RelationPairs(sibling=array(sibling), gene=array(gene_pairs))


def split_pairs(pairs: np.ndarray, split_by_index: np.ndarray) -> dict[str, np.ndarray]:
    """Train requires two train endpoints; test endpoints never provide labels."""

    if not len(pairs):
        empty = np.empty((0, 2), dtype=np.int64)
        return {"train": empty, "validation": empty, "test": empty}
    a, b = split_by_index[pairs[:, 0]], split_by_index[pairs[:, 1]]
    train = (a == "train") & (b == "train")
    test = (a == "test") | (b == "test")
    validation = ~(train | test)
    return {
        "train": pairs[train],
        "validation": pairs[validation],
        "test": pairs[test],
    }


def _known_neighbours(pairs: np.ndarray, n: int) -> list[set[int]]:
    known = [set() for _ in range(n)]
    for a, b in pairs:
        known[int(a)].add(int(b))
        known[int(b)].add(int(a))
    return known


def _sample_negatives(
    positive: np.ndarray,
    count: int,
    candidates: np.ndarray,
    known: list[set[int]],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not len(positive):
        empty = np.array([], dtype=np.int64)
        return empty, empty, empty
    oriented = positive.copy()
    swap = rng.random(len(oriented)) < 0.5
    oriented[swap] = oriented[swap, ::-1]
    anchors = np.repeat(oriented[:, 0], count)
    partners = np.repeat(oriented[:, 1], count)
    negatives = candidates[rng.integers(0, len(candidates), len(anchors))]
    bad = np.array(
        [negative == anchor or int(negative) in known[int(anchor)] for anchor, negative in zip(anchors, negatives)]
    )
    for _ in range(20):
        if not bad.any():
            break
        negatives[bad] = candidates[rng.integers(0, len(candidates), int(bad.sum()))]
        bad = np.array(
            [negative == anchor or int(negative) in known[int(anchor)] for anchor, negative in zip(anchors, negatives)]
        )
    keep = ~bad
    return anchors[keep], partners[keep], negatives[keep]


@torch.no_grad()
def _validation_auc(
    net: EmbeddingNet,
    inputs: dict[str, torch.Tensor],
    available: torch.Tensor,
    pairs: np.ndarray,
    all_pairs: np.ndarray,
    task: str,
    candidates: np.ndarray,
    negatives: int,
    rng: np.random.Generator,
) -> float:
    if not len(pairs):
        return float("nan")
    net.eval()
    mask = torch.tensor([name in TASK_MASKS[task] for name in MODALITIES], dtype=torch.bool, device=available.device)
    embeddings, _ = net.encode(inputs, available, mask)
    known = _known_neighbours(all_pairs, len(available))
    anchors, partners, negative = _sample_negatives(pairs, negatives, candidates, known, rng)
    positive_scores = (embeddings[anchors] * embeddings[partners]).sum(-1).cpu().numpy()
    negative_scores = (embeddings[anchors] * embeddings[negative]).sum(-1).cpu().numpy()
    y = np.r_[np.ones(len(positive_scores)), np.zeros(len(negative_scores))]
    scores = np.r_[positive_scores, negative_scores]
    return float(roc_auc_score(y, scores))


def train_network(
    matrices: dict[str, sp.csr_matrix],
    available: np.ndarray,
    relations: RelationPairs,
    split_by_index: np.ndarray,
    config: Optional[dict[str, Any]] = None,
    verbose: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    config = {**DEFAULT_CONFIG, **(config or {})}
    seed = int(config["seed"])
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    target = device()
    inputs = {name: to_torch(matrices[name], target) for name in MODALITIES}
    availability = torch.from_numpy(available.astype(bool)).to(target)
    dimensions = {name: matrices[name].shape[1] for name in MODALITIES}
    net = EmbeddingNet(dimensions, config).to(target)
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    pools = {task: split_pairs(pairs, split_by_index) for task, pairs in relations.by_task().items()}
    train_candidates = np.flatnonzero(split_by_index == "train")
    validation_candidates = np.flatnonzero(split_by_index != "test")
    known = {
        task: _known_neighbours(pairs, len(split_by_index))
        for task, pairs in relations.by_task().items()
    }
    masks = {
        task: torch.tensor(
            [name in TASK_MASKS[task] for name in MODALITIES], dtype=torch.bool, device=target
        )
        for task in TASK_MASKS
    }
    best_score, best_state, bad_epochs = -np.inf, None, 0
    history: list[dict[str, Any]] = []
    max_pairs = int(config["max_pairs_per_task"])
    for epoch in range(int(config["epochs"])):
        net.train()
        loss = torch.zeros((), device=target)
        task_losses: dict[str, float] = {}
        for task in ("sibling", "gene"):
            positives = pools[task]["train"]
            if len(positives) > max_pairs:
                positives = positives[rng.choice(len(positives), max_pairs, replace=False)]
            anchors, partners, negatives = _sample_negatives(
                positives,
                int(config["negatives"]),
                train_candidates,
                known[task],
                rng,
            )
            embeddings, _ = net.encode(
                inputs, availability, masks[task], float(config["modality_dropout"])
            )
            positive_score = (embeddings[anchors] * embeddings[partners]).sum(-1)
            negative_score = (embeddings[anchors] * embeddings[negatives]).sum(-1)
            task_loss = F.softplus(
                (negative_score - positive_score + float(config["margin"])) / float(config["temperature"])
            ).mean()
            loss = loss + task_loss
            task_losses[task] = float(task_loss.detach())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        aucs = {
            task: _validation_auc(
                net,
                inputs,
                availability,
                pools[task]["validation"],
                relations.by_task()[task],
                task,
                validation_candidates,
                int(config["validation_negatives"]),
                np.random.default_rng(seed + epoch * 10 + offset),
            )
            for offset, task in enumerate(("sibling", "gene"))
        }
        score = float(np.nanmean(list(aucs.values())))
        row = {
            "epoch": epoch + 1,
            "loss": float(loss.detach()),
            **{f"{task}_loss": value for task, value in task_losses.items()},
            **{f"validation_{task}_auc": value for task, value in aucs.items()},
            "validation_mean_auc": score,
        }
        history.append(row)
        if verbose:
            print(
                f"[v4] epoch {epoch + 1:02d} loss={row['loss']:.4f} "
                f"validation_auc={score:.4f}",
                flush=True,
            )
        if score > best_score + 1e-4:
            best_score, bad_epochs = score, 0
            best_state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= int(config["patience"]):
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a finite validation score")
    net.load_state_dict(best_state)

    test_candidates = np.arange(len(split_by_index))
    test_auc = {
        task: _validation_auc(
            net,
            inputs,
            availability,
            pools[task]["test"],
            relations.by_task()[task],
            task,
            test_candidates,
            int(config["validation_negatives"]),
            np.random.default_rng(seed + 10_000 + offset),
        )
        for offset, task in enumerate(("sibling", "gene"))
    }
    diagnostics = {
        "device": str(target),
        "best_epoch": int(np.argmax([row["validation_mean_auc"] for row in history])) + 1,
        "best_validation_mean_auc": best_score,
        "test_auc": test_auc,
        "pair_counts": {
            task: {split: len(values) for split, values in task_pools.items()}
            for task, task_pools in pools.items()
        },
    }
    return best_state, history, diagnostics


@dataclass
class EmbeddingModel:
    ids: list[str]
    names: list[str]
    encoder: DiseaseEncoder
    matrices: dict[str, sp.csr_matrix]
    available: np.ndarray
    state: dict[str, Any]
    config: dict[str, Any]
    embeddings_full: np.ndarray
    history: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    _net: Optional[EmbeddingNet] = field(default=None, repr=False)
    _embedding_cache: dict[tuple[str, ...], np.ndarray] = field(default_factory=dict, repr=False)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_net"] = None
        state["_embedding_cache"] = {}
        return state

    @property
    def net(self) -> EmbeddingNet:
        if self._net is None:
            dimensions = {name: self.matrices[name].shape[1] for name in MODALITIES}
            net = EmbeddingNet(dimensions, self.config)
            net.load_state_dict(self.state)
            self._net = net.to(device()).eval()
        return self._net

    @torch.no_grad()
    def embeddings(self, masked: Sequence[str] = ()) -> np.ndarray:
        key = tuple(sorted(masked))
        if not key:
            return self.embeddings_full
        if key not in self._embedding_cache:
            target = device()
            inputs = {name: to_torch(self.matrices[name], target) for name in MODALITIES}
            available = torch.from_numpy(self.available.astype(bool)).to(target)
            mask = torch.tensor([name in key for name in MODALITIES], dtype=torch.bool, device=target)
            values, _ = self.net.encode(inputs, available, mask)
            self._embedding_cache[key] = values.cpu().numpy().astype(np.float32)
        return self._embedding_cache[key]

    def score_queries(self, query_ids: Sequence[str], masked: Sequence[str] = ()) -> np.ndarray:
        index = {disease: position for position, disease in enumerate(self.ids)}
        missing = sorted(set(query_ids) - set(index))
        if missing:
            raise KeyError(f"Queries absent from the v4 catalogue: {missing[:5]}")
        embeddings = self.embeddings(masked)
        rows = np.array([index[disease] for disease in query_ids], dtype=np.int64)
        return np.asarray(embeddings[rows] @ embeddings.T, dtype=np.float32)

    def save(self) -> None:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, MODEL_PATH, compress=3)


@torch.no_grad()
def encode_full(
    matrices: dict[str, sp.csr_matrix],
    available: np.ndarray,
    state: dict[str, Any],
    config: dict[str, Any],
) -> np.ndarray:
    target = device()
    dimensions = {name: matrices[name].shape[1] for name in MODALITIES}
    net = EmbeddingNet(dimensions, config).to(target)
    net.load_state_dict(state)
    net.eval()
    inputs = {name: to_torch(matrices[name], target) for name in MODALITIES}
    values, _ = net.encode(
        inputs,
        torch.from_numpy(available.astype(bool)).to(target),
        torch.zeros(len(MODALITIES), dtype=torch.bool, device=target),
    )
    return values.cpu().numpy().astype(np.float32)


def load_model() -> EmbeddingModel:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"{MODEL_PATH} not found. Run `python v4/train.py` first.")
    return joblib.load(MODEL_PATH)
