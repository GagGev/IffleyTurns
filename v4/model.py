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
    "temperature": 0.10,
    "random_negatives": 48,
    "hard_negatives": 16,
    "hard_candidate_buffer": 64,
    "max_pairs_per_task": 4_096,
    "validation_negatives": 20,
    "phenotype_neighbours": 3,
    "phenotype_min_similarity": 0.35,
    "fusion_C": 0.03,
    "fusion_negatives": 8,
    "fusion_mixes": (0.0, 0.25, 0.5, 0.75, 1.0),
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
    phenotype: np.ndarray

    def by_task(self) -> dict[str, np.ndarray]:
        return {
            "sibling": self.sibling,
            "gene": self.gene,
            "phenotype": self.phenotype,
        }


def phenotype_pairs(
    matrix: sp.csr_matrix,
    neighbours: int = 3,
    min_similarity: float = 0.35,
    block_size: int = 256,
) -> np.ndarray:
    """Top HPO-cosine neighbours used only as an auxiliary relation label."""

    matrix = matrix.tocsr()
    pairs: set[tuple[int, int]] = set()
    if neighbours <= 0:
        return np.empty((0, 2), dtype=np.int64)
    for start in range(0, matrix.shape[0], block_size):
        scores = (matrix[start : start + block_size] @ matrix.T).toarray()
        for local, row in enumerate(scores):
            anchor = start + local
            row[anchor] = -np.inf
            count = min(neighbours, len(row) - 1)
            selected = np.argpartition(row, -count)[-count:]
            for partner in selected:
                if row[partner] >= min_similarity:
                    pairs.add((min(anchor, int(partner)), max(anchor, int(partner))))
    return np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2)


def build_relation_pairs(
    bundle: Bundle,
    ids: Sequence[str],
    phenotype_matrix: Optional[sp.csr_matrix] = None,
    phenotype_neighbours: int = 3,
    phenotype_min_similarity: float = 0.35,
) -> RelationPairs:
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

    phenotype = (
        phenotype_pairs(phenotype_matrix, phenotype_neighbours, phenotype_min_similarity)
        if phenotype_matrix is not None
        else np.empty((0, 2), dtype=np.int64)
    )
    return RelationPairs(sibling=array(sibling), gene=array(gene_pairs), phenotype=phenotype)


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


def union_known_neighbours(relations: RelationPairs, n: int) -> list[set[int]]:
    """Relations from any auxiliary task are never negatives for another."""

    known = [set() for _ in range(n)]
    for pairs in relations.by_task().values():
        for a, b in pairs:
            known[int(a)].add(int(b))
            known[int(b)].add(int(a))
    return known


def sample_negative_matrix(
    anchors: np.ndarray,
    count: int,
    candidates: np.ndarray,
    known: list[set[int]],
    rng: np.random.Generator,
) -> np.ndarray:
    """Uniform negatives, excluding self and every known auxiliary positive."""

    result = np.empty((len(anchors), count), dtype=np.int64)
    if count == 0:
        return result
    for row, anchor in enumerate(anchors):
        excluded = known[int(anchor)] | {int(anchor)}
        filled = 0
        while filled < count:
            draw = candidates[rng.integers(0, len(candidates), max(2 * (count - filled), 8))]
            valid = [int(value) for value in draw if int(value) not in excluded]
            take = min(len(valid), count - filled)
            if take:
                result[row, filled : filled + take] = valid[:take]
                filled += take
    return result


@torch.no_grad()
def hard_negative_matrix(
    embeddings: torch.Tensor,
    anchors: np.ndarray,
    count: int,
    candidates: np.ndarray,
    known: list[set[int]],
    rng: np.random.Generator,
    buffer: int = 64,
    block_size: int = 256,
) -> np.ndarray:
    """Current nearest non-positive train diseases for each anchor."""

    if count == 0:
        return np.empty((len(anchors), 0), dtype=np.int64)
    unique, inverse = np.unique(anchors, return_inverse=True)
    candidate_tensor = torch.from_numpy(candidates).to(embeddings.device)
    selected = np.empty((len(unique), count), dtype=np.int64)
    width = min(len(candidates), count + buffer)
    for start in range(0, len(unique), block_size):
        block = unique[start : start + block_size]
        block_tensor = torch.from_numpy(block).to(embeddings.device)
        scores = embeddings[block_tensor] @ embeddings[candidate_tensor].T
        top = torch.topk(scores, width, dim=1).indices.cpu().numpy()
        for local, anchor in enumerate(block):
            ordered = candidates[top[local]]
            valid = [
                int(value)
                for value in ordered
                if int(value) != int(anchor) and int(value) not in known[int(anchor)]
            ]
            if len(valid) < count:
                fallback = sample_negative_matrix(
                    np.array([anchor]), count - len(valid), candidates, known, rng
                )[0].tolist()
                valid.extend(fallback)
            selected[start + local] = valid[:count]
    return selected[inverse]


def sampled_softmax_loss(
    embeddings: torch.Tensor,
    anchors: np.ndarray,
    partners: np.ndarray,
    negatives: np.ndarray,
    temperature: float,
) -> torch.Tensor:
    """InfoNCE with the positive in column zero and cosine-only logits."""

    target = embeddings.device
    anchor_index = torch.from_numpy(anchors).to(target)
    partner_index = torch.from_numpy(partners).to(target)
    negative_index = torch.from_numpy(negatives).to(target)
    anchor_embedding = embeddings[anchor_index]
    positive = (anchor_embedding * embeddings[partner_index]).sum(-1, keepdim=True)
    negative = (anchor_embedding[:, None, :] * embeddings[negative_index]).sum(-1)
    logits = torch.cat([positive, negative], dim=1) / temperature
    labels = torch.zeros(len(anchors), dtype=torch.long, device=target)
    return F.cross_entropy(logits, labels)


@torch.no_grad()
def retrieval_metrics(
    embeddings: torch.Tensor,
    pairs: np.ndarray,
    queries: np.ndarray,
    gallery: np.ndarray,
) -> dict[str, float]:
    """Macro retrieval metrics for relation-positive partners in a gallery."""

    neighbours = _known_neighbours(pairs, len(embeddings))
    gallery_set = set(int(value) for value in gallery)
    eligible = np.array(
        [
            int(query)
            for query in queries
            if any(partner in gallery_set and partner != int(query) for partner in neighbours[int(query)])
        ],
        dtype=np.int64,
    )
    if not len(eligible):
        return {"map": float("nan"), "mrr": float("nan"), "hits@10": float("nan"), "n_queries": 0}
    query_tensor = torch.from_numpy(eligible).to(embeddings.device)
    gallery_tensor = torch.from_numpy(gallery).to(embeddings.device)
    scores = (embeddings[query_tensor] @ embeddings[gallery_tensor].T).cpu().numpy()
    gallery_index = {int(value): index for index, value in enumerate(gallery)}
    aps, reciprocal, hits = [], [], []
    for row, query in enumerate(eligible):
        own = gallery_index.get(int(query))
        if own is not None:
            scores[row, own] = -np.inf
        positives = {
            gallery_index[partner]
            for partner in neighbours[int(query)]
            if partner in gallery_index and partner != int(query)
        }
        order = np.argsort(-scores[row])
        rank_by_column = np.empty(len(order), dtype=np.int64)
        rank_by_column[order] = np.arange(1, len(order) + 1)
        ranks = np.sort(rank_by_column[list(positives)])
        aps.append(float(np.mean(np.arange(1, len(ranks) + 1) / ranks)))
        reciprocal.append(1.0 / ranks[0])
        hits.append(float(ranks[0] <= 10))
    return {
        "map": float(np.mean(aps)),
        "mrr": float(np.mean(reciprocal)),
        "hits@10": float(np.mean(hits)),
        "n_queries": len(eligible),
    }


def retrieval_from_scores(
    scores: np.ndarray,
    pairs: np.ndarray,
    queries: np.ndarray,
    gallery: np.ndarray,
) -> dict[str, float]:
    """Same retrieval metrics as ``retrieval_metrics``, from an explicit score matrix."""

    neighbours = _known_neighbours(pairs, scores.shape[1])
    gallery_set = set(int(value) for value in gallery)
    eligible = [
        (row, int(query))
        for row, query in enumerate(queries)
        if any(partner in gallery_set and partner != int(query) for partner in neighbours[int(query)])
    ]
    if not eligible:
        return {"map": float("nan"), "mrr": float("nan"), "hits@10": float("nan"), "n_queries": 0}
    gallery_index = {int(value): index for index, value in enumerate(gallery)}
    aps, reciprocal, hits = [], [], []
    for row, query in eligible:
        row_scores = scores[row].copy()
        own = gallery_index.get(query)
        if own is not None:
            row_scores[query] = -np.inf
        positives = {
            gallery_index[partner]
            for partner in neighbours[query]
            if partner in gallery_index and partner != query
        }
        ranked = row_scores[gallery]
        order = np.argsort(-ranked)
        rank_by_column = np.empty(len(order), dtype=np.int64)
        rank_by_column[order] = np.arange(1, len(order) + 1)
        ranks = np.sort(rank_by_column[list(positives)])
        aps.append(float(np.mean(np.arange(1, len(ranks) + 1) / ranks)))
        reciprocal.append(1.0 / ranks[0])
        hits.append(float(ranks[0] <= 10))
    return {
        "map": float(np.mean(aps)),
        "mrr": float(np.mean(reciprocal)),
        "hits@10": float(np.mean(hits)),
        "n_queries": len(eligible),
    }


def choose_fusion_mix(
    engine,
    fusion,
    embeddings_by_task: dict[str, np.ndarray],
    relations: RelationPairs,
    split_by_index: np.ndarray,
    embedding_scale: float,
    fusion_scale: float,
    mixes: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> tuple[float, dict[str, float]]:
    """Pick the fusion mix on validation MAP for sibling and gene retrieval."""

    from fusion import combine_scores

    queries = np.flatnonzero(split_by_index == "validation")
    gallery = np.flatnonzero(split_by_index != "test")
    best_mix, best_score, details = 0.5, -np.inf, {}
    for mix in mixes:
        maps = []
        for task in ("sibling", "gene"):
            embeddings = embeddings_by_task[task]
            S, A = engine.block(queries, masked=TASK_MASKS[task])
            scores = combine_scores(
                embeddings[queries] @ embeddings.T,
                fusion.score(S, A),
                embedding_scale,
                fusion_scale,
                float(mix),
            )
            maps.append(retrieval_from_scores(scores, relations.by_task()[task], queries, gallery)["map"])
        score = float(np.nanmean(maps))
        details[str(mix)] = score
        if score > best_score:
            best_mix, best_score = float(mix), score
    return best_mix, details


def best_retrieval_epoch(history: Sequence[dict[str, Any]]) -> int:
    """Zero-based checkpoint index selected only by validation retrieval MAP."""

    return int(np.nanargmax([row["validation_mean_map"] for row in history]))


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
    validation_queries = np.flatnonzero(split_by_index == "validation")
    known = union_known_neighbours(relations, len(split_by_index))
    masks = {
        task: torch.tensor(
            [name in TASK_MASKS[task] for name in MODALITIES], dtype=torch.bool, device=target
        )
        for task in TASK_MASKS
    }
    tasks = tuple(relations.by_task())
    best_score, best_state, bad_epochs = -np.inf, None, 0
    history: list[dict[str, Any]] = []
    max_pairs = int(config["max_pairs_per_task"])
    for epoch in range(int(config["epochs"])):
        net.train()
        loss = torch.zeros((), device=target)
        task_losses: dict[str, float] = {}
        for task in tasks:
            positives = pools[task]["train"]
            if len(positives) > max_pairs:
                positives = positives[rng.choice(len(positives), max_pairs, replace=False)]
            oriented = positives.copy()
            swap = rng.random(len(oriented)) < 0.5
            oriented[swap] = oriented[swap, ::-1]
            anchors, partners = oriented[:, 0], oriented[:, 1]
            embeddings, _ = net.encode(
                inputs, availability, masks[task], float(config["modality_dropout"])
            )
            random_negatives = sample_negative_matrix(
                anchors,
                int(config["random_negatives"]),
                train_candidates,
                known,
                rng,
            )
            hard_negatives = hard_negative_matrix(
                embeddings.detach(),
                anchors,
                int(config["hard_negatives"]),
                train_candidates,
                known,
                rng,
                int(config["hard_candidate_buffer"]),
            )
            negatives = np.concatenate([hard_negatives, random_negatives], axis=1)
            task_loss = sampled_softmax_loss(
                embeddings,
                anchors,
                partners,
                negatives,
                float(config["temperature"]),
            )
            loss = loss + task_loss
            task_losses[task] = float(task_loss.detach())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        validation_retrieval, aucs = {}, {}
        for offset, task in enumerate(tasks):
            net.eval()
            with torch.no_grad():
                embeddings, _ = net.encode(inputs, availability, masks[task])
            validation_retrieval[task] = retrieval_metrics(
                embeddings,
                relations.by_task()[task],
                validation_queries,
                validation_candidates,
            )
            aucs[task] = _validation_auc(
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
        score = float(np.nanmean([entry["map"] for entry in validation_retrieval.values()]))
        row = {
            "epoch": epoch + 1,
            "loss": float(loss.detach()),
            **{f"{task}_loss": value for task, value in task_losses.items()},
            **{f"validation_{task}_auc": value for task, value in aucs.items()},
            **{
                f"validation_{task}_{metric}": value
                for task, entry in validation_retrieval.items()
                for metric, value in entry.items()
            },
            "validation_mean_auc": float(np.nanmean(list(aucs.values()))),
            "validation_mean_map": score,
        }
        history.append(row)
        if verbose:
            print(
                f"[v4] epoch {epoch + 1:02d} loss={row['loss']:.4f} "
                f"validation_map={score:.4f} validation_auc={row['validation_mean_auc']:.4f}",
                flush=True,
            )
        if score > best_score + 1e-5:
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
    test_queries = np.flatnonzero(split_by_index == "test")
    test_auc, test_retrieval = {}, {}
    for offset, task in enumerate(tasks):
        net.eval()
        with torch.no_grad():
            embeddings, _ = net.encode(inputs, availability, masks[task])
        test_retrieval[task] = retrieval_metrics(
            embeddings,
            relations.by_task()[task],
            test_queries,
            test_candidates,
        )
        test_auc[task] = _validation_auc(
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
    best_index = best_retrieval_epoch(history)
    diagnostics = {
        "device": str(target),
        "selection_metric": "validation_mean_map",
        "best_epoch": best_index + 1,
        "best_validation_mean_map": best_score,
        "best_validation_mean_auc": history[best_index]["validation_mean_auc"],
        "test_auc": test_auc,
        "test_retrieval": test_retrieval,
        "pair_counts": {
            task: {split: len(values) for split, values in task_pools.items()}
            for task, task_pools in pools.items()
        },
        "leakage_checks": {
            task: bool(
                all(split_by_index[index] == "train" for pair in task_pools["train"] for index in pair)
            )
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
    fusion: Any = None
    embedding_scale: float = 1.0
    fusion_scale: float = 1.0
    fusion_mix: float = 0.0
    _engine: Any = field(default=None, repr=False)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_net"] = None
        state["_embedding_cache"] = {}
        state["_engine"] = None
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
        embedding_scores = np.asarray(embeddings[rows] @ embeddings.T, dtype=np.float32)
        if self.fusion is None or self.fusion_mix == 0:
            return embedding_scores
        from fusion import CosineEngine, combine_scores

        if self._engine is None:
            self._engine = CosineEngine(self.matrices, self.available)
        S, A = self._engine.block(rows, masked=masked)
        return combine_scores(
            embedding_scores,
            self.fusion.score(S, A),
            self.embedding_scale,
            self.fusion_scale,
            self.fusion_mix,
        )

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


@torch.no_grad()
def encode_masked(
    matrices: dict[str, sp.csr_matrix],
    available: np.ndarray,
    state: dict[str, Any],
    config: dict[str, Any],
    masked: Sequence[str] = (),
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
        torch.tensor([name in masked for name in MODALITIES], dtype=torch.bool, device=target),
    )
    return values.cpu().numpy().astype(np.float32)


def load_model() -> EmbeddingModel:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"{MODEL_PATH} not found. Run `python v4/train.py` first.")
    return joblib.load(MODEL_PATH)
