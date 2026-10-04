"""v2-style non-negative fusion of per-modality cosines, trained inside v4.

The fusion is a leakage-safe copy of v2's logistic: it scores pairs from the
same nine drug-free sparse vectors the embedding uses, with the same task
masks.  v4 never imports ``v2/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize

from modalities import MODALITIES, TASK_MASKS


@dataclass
class LinearFusion:
    modalities: tuple[str, ...]
    similarity_weights: np.ndarray
    availability_weights: np.ndarray
    intercept: float
    params: dict[str, Any] = field(default_factory=dict)

    def score(self, S: np.ndarray, A: np.ndarray) -> np.ndarray:
        return (
            np.tensordot(S, self.similarity_weights, axes=([S.ndim - 1], [0]))
            + np.tensordot(A.astype(np.float32), self.availability_weights, axes=([A.ndim - 1], [0]))
            + self.intercept
        )


class CosineEngine:
    """Query-versus-catalogue and pair-wise cosine blocks."""

    def __init__(self, matrices: dict[str, sp.csr_matrix], available: np.ndarray):
        self.modalities = tuple(MODALITIES)
        self.matrices = {name: matrices[name].tocsr() for name in self.modalities}
        self.available = np.asarray(available, dtype=bool)
        self.transposed = {name: matrix.T.tocsr() for name, matrix in self.matrices.items()}

    def mask_vector(self, masked: Sequence[str]) -> np.ndarray:
        return np.array([name in masked for name in self.modalities], dtype=bool)

    def pairs(self, a: np.ndarray, b: np.ndarray, masked: Sequence[str] = ()) -> tuple[np.ndarray, np.ndarray]:
        mask = self.mask_vector(masked)
        S = np.zeros((len(a), len(self.modalities)), dtype=np.float32)
        for column, name in enumerate(self.modalities):
            if mask[column]:
                continue
            S[:, column] = np.asarray(self.matrices[name][a].multiply(self.matrices[name][b]).sum(axis=1)).ravel()
        A = self.available[a] & self.available[b]
        A[:, mask] = False
        S[~A] = 0.0
        return S, A

    def block(self, rows: np.ndarray, masked: Sequence[str] = ()) -> tuple[np.ndarray, np.ndarray]:
        mask = self.mask_vector(masked)
        n_q, n = len(rows), self.available.shape[0]
        S = np.zeros((n_q, n, len(self.modalities)), dtype=np.float32)
        for column, name in enumerate(self.modalities):
            if mask[column]:
                continue
            product = self.matrices[name][rows] @ self.transposed[name]
            S[:, :, column] = product.toarray() if sp.issparse(product) else product
        A = self.available[rows][:, None, :] & self.available[None, :, :]
        A[:, :, mask] = False
        S[~A] = 0.0
        return S, A


def _bounded_logistic(X: np.ndarray, y: np.ndarray, weight: np.ndarray, C: float, n_nonnegative: int) -> tuple[np.ndarray, float]:
    total = float(weight.sum())
    y = y.astype(np.float64)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        w, b = theta[:-1], theta[-1]
        z = X @ w + b
        loss = np.logaddexp(0.0, z) - y * z
        residual = weight * (1.0 / (1.0 + np.exp(-z)) - y)
        value = (weight @ loss) / total + (w @ w) / (2.0 * C * total)
        gradient = np.r_[X.T @ residual / total + w / (C * total), residual.sum() / total]
        return value, gradient

    bounds = [(0.0, None)] * n_nonnegative + [(None, None)] * (X.shape[1] - n_nonnegative + 1)
    result = minimize(
        objective,
        np.zeros(X.shape[1] + 1),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 5000},
    )
    return result.x[:-1], float(result.x[-1])


def _sample_fusion_pairs(
    positives: np.ndarray,
    candidates: np.ndarray,
    known: list[set[int]],
    negatives: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not len(positives):
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty
    oriented = positives.copy()
    swap = rng.random(len(oriented)) < 0.5
    oriented[swap] = oriented[swap, ::-1]
    anchors = np.repeat(oriented[:, 0], negatives)
    partners = np.repeat(oriented[:, 1], negatives)
    drawn = candidates[rng.integers(0, len(candidates), len(anchors))]
    keep = np.array(
        [int(negative) != int(anchor) and int(negative) not in known[int(anchor)] for anchor, negative in zip(anchors, drawn)]
    )
    return (
        np.r_[oriented[:, 0], anchors[keep]],
        np.r_[oriented[:, 1], drawn[keep]],
        np.r_[np.ones(len(oriented)), np.zeros(int(keep.sum()))],
    )


def fit_fusion(
    engine: CosineEngine,
    train_pairs: dict[str, np.ndarray],
    train_candidates: np.ndarray,
    known: list[set[int]],
    config: Optional[dict[str, Any]] = None,
) -> LinearFusion:
    """Non-negative logistic fusion on train/train pairs with the same task masks."""

    config = config or {}
    rng = np.random.default_rng(int(config.get("seed", 0)))
    max_pairs = int(config.get("max_pairs_per_task", 4_096))
    negatives = int(config.get("fusion_negatives", 8))
    parts_S, parts_A, parts_y, parts_w = [], [], [], []
    for task, positives in train_pairs.items():
        if len(positives) > max_pairs:
            positives = positives[rng.choice(len(positives), max_pairs, replace=False)]
        a, b, y = _sample_fusion_pairs(positives, train_candidates, known, negatives, rng)
        if not len(a):
            continue
        S, A = engine.pairs(a, b, masked=TASK_MASKS[task])
        weight = np.full(len(y), 1.0 / max(len(y), 1))
        parts_S.append(S)
        parts_A.append(A)
        parts_y.append(y.astype(np.float64))
        parts_w.append(weight)
    S = np.concatenate(parts_S)
    A = np.concatenate(parts_A)
    y = np.concatenate(parts_y)
    weight = np.concatenate(parts_w)
    weight = weight * len(weight) / weight.sum()
    X = np.hstack([S, A.astype(np.float32)])
    coef, intercept = _bounded_logistic(X.astype(np.float64), y, weight, float(config.get("fusion_C", 0.03)), len(MODALITIES))
    return LinearFusion(
        modalities=tuple(MODALITIES),
        similarity_weights=np.asarray(coef[: len(MODALITIES)], dtype=np.float32),
        availability_weights=np.asarray(coef[len(MODALITIES) :], dtype=np.float32),
        intercept=intercept,
        params={"C": float(config.get("fusion_C", 0.03)), "nonnegative": True},
    )


def combine_scores(
    embedding: np.ndarray,
    fusion: np.ndarray,
    embedding_scale: float,
    fusion_scale: float,
    mix: float,
) -> np.ndarray:
    """``mix`` is the weight on the v2-style fusion after both scores are scaled."""

    embedding_scale = embedding_scale or 1.0
    fusion_scale = fusion_scale or 1.0
    return ((1.0 - mix) * (embedding / embedding_scale) + mix * (fusion / fusion_scale)).astype(np.float32)


def pair_score_scales(
    embeddings: np.ndarray,
    engine: CosineEngine,
    fusion: LinearFusion,
    a: np.ndarray,
    b: np.ndarray,
) -> tuple[float, float]:
    embedding = (embeddings[a] * embeddings[b]).sum(-1)
    S, A = engine.pairs(a, b)
    fused = fusion.score(S, A)
    return float(np.std(embedding) or 1.0), float(np.std(fused) or 1.0)
