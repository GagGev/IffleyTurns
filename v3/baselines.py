"""Baselines: v1 weighted Jaccard, v2 fusion (as shipped and retrained), GBM over cosines."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier

from common import V3_DATA_DIR
from modalities import MODALITIES

V1_WEIGHTS = {
    "phenotypes": 0.35,
    "genes": 0.20,
    "classifications": 0.15,
    "body_systems": 0.10,
    "inheritance": 0.07,
    "onset": 0.05,
    "prevalence": 0.05,
    "approved_drugs": 0.03,
}
V1_COLUMNS = {
    "phenotypes": "hpo_ids",
    "genes": "gene_symbols",
    "classifications": "category_ids",
    "body_systems": "body_system_ids",
    "inheritance": "inheritance",
    "onset": "onset",
    "approved_drugs": "approved_drug_ids",
}
# The v1 drug family is today's approved-drug list: undated, so it is hidden.
V1_MASKED = ("approved_drugs",)
V2_FUSION_PATH = V3_DATA_DIR / "v2_production_fusion.json"


class V1Baseline:
    """The v1 metric: weighted Jaccard over raw annotation sets plus
    exp(-|log10 prevalence difference|), renormalized over available families."""

    name = "v1_weighted_jaccard"

    def __init__(self, v1_sets: dict[str, dict[str, Any]], ids: Sequence[str]):
        self.ids = list(ids)
        self.binary: dict[str, sp.csr_matrix] = {}
        for family, column in V1_COLUMNS.items():
            vocabulary: dict[str, int] = {}
            rows, cols = [], []
            for row, disease in enumerate(self.ids):
                for item in v1_sets[disease][column]:
                    cols.append(vocabulary.setdefault(item, len(vocabulary)))
                    rows.append(row)
            self.binary[family] = sp.csr_matrix(
                (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(self.ids), max(len(vocabulary), 1))
            )
        self.transposed = {f: x.T.tocsr() for f, x in self.binary.items()}
        self.sizes = {f: np.asarray(x.sum(axis=1)).ravel() for f, x in self.binary.items()}
        prevalence = [v1_sets[d]["prevalence"] for d in self.ids]
        self.log_prevalence = np.array([math.log10(p) if p else np.nan for p in prevalence])

    def block(self, rows: np.ndarray, masked: Sequence[str] = V1_MASKED) -> np.ndarray:
        numerator = np.zeros((len(rows), len(self.ids)))
        denominator = np.zeros_like(numerator)
        for family, weight in V1_WEIGHTS.items():
            if family in masked:
                continue
            if family == "prevalence":
                lq, lg = self.log_prevalence[rows][:, None], self.log_prevalence[None, :]
                available = ~np.isnan(lq) & ~np.isnan(lg)
                similarity = np.where(available, np.exp(-np.abs(np.nan_to_num(lq) - np.nan_to_num(lg))), 0.0)
            else:
                intersection = (self.binary[family][rows] @ self.transposed[family]).toarray()
                sq, sg = self.sizes[family][rows][:, None], self.sizes[family][None, :]
                available = (sq > 0) & (sg > 0)
                union = np.maximum(sq + sg - intersection, 1.0)
                similarity = np.where(available, intersection / union, 0.0)
            numerator += weight * similarity
            denominator += weight * available
        return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0).astype(np.float32)


@dataclass
class LinearFusion:
    """logit = b + sum_m (w_m * sim_m + v_m * available_m) over the static modalities."""

    similarity_weights: np.ndarray
    availability_weights: np.ndarray
    intercept: float
    name: str = "logistic_fusion"

    def score(self, S: np.ndarray, A: np.ndarray) -> np.ndarray:
        return (
            np.tensordot(S, self.similarity_weights, axes=([-1], [0]))
            + np.tensordot(A.astype(np.float32), self.availability_weights, axes=([-1], [0]))
            + self.intercept
        ).astype(np.float32)

    def coefficients(self) -> list[dict[str, Any]]:
        return [
            {"modality": m, "similarity_weight": float(w), "availability_weight": float(v)}
            for m, w, v in zip(MODALITIES, self.similarity_weights, self.availability_weights)
        ]


def v2_shipped() -> Optional[LinearFusion]:
    """v2's production fusion with its drug-derived modalities hidden."""

    if not V2_FUSION_PATH.is_file():
        return None
    stored = json.loads(V2_FUSION_PATH.read_text(encoding="utf-8"))
    weights = dict(zip(stored["modalities"], stored["similarity_weights"]))
    availability = dict(zip(stored["modalities"], stored["availability_weights"]))
    return LinearFusion(
        similarity_weights=np.array([weights.get(m, 0.0) for m in MODALITIES], dtype=np.float32),
        availability_weights=np.array([availability.get(m, 0.0) for m in MODALITIES], dtype=np.float32),
        intercept=float(stored["intercept"]),
        name="v2_shipped",
    )


def fit_nonnegative_logistic(S: np.ndarray, A: np.ndarray, y: np.ndarray, C: float = 0.03) -> LinearFusion:
    """v2's fusion recipe: class-balanced L2 logistic regression with
    non-negative similarity weights (more similarity never lowers a score)."""

    X = np.hstack([S, A.astype(np.float32)]).astype(np.float64)
    n = S.shape[1]
    rate = y.mean()
    weight = np.where(y > 0, 0.5 / rate, 0.5 / (1 - rate))
    total = weight.sum()

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        w, b = theta[:-1], theta[-1]
        z = X @ w + b
        loss = np.logaddexp(0.0, z) - y * z
        residual = weight * (1.0 / (1.0 + np.exp(-z)) - y)
        value = (weight @ loss) / total + (w @ w) / (2.0 * C * total)
        gradient = np.r_[X.T @ residual / total + w / (C * total), residual.sum() / total]
        return value, gradient

    bounds = [(0.0, None)] * n + [(None, None)] * (X.shape[1] - n + 1)
    result = minimize(objective, np.zeros(X.shape[1] + 1), jac=True, method="L-BFGS-B", bounds=bounds)
    coef, intercept = result.x[:-1], float(result.x[-1])
    return LinearFusion(
        similarity_weights=coef[:n].astype(np.float32),
        availability_weights=coef[n:].astype(np.float32),
        intercept=intercept,
        name="v2_retrained",
    )


class GBMStatic:
    """Gradient-boosted trees over the static cosines (NaN when unavailable)."""

    name = "gbm_static"

    def __init__(self, seed: int = 0):
        self.model = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=seed
        )

    @staticmethod
    def features(S: np.ndarray, A: np.ndarray) -> np.ndarray:
        return np.concatenate([np.where(A, S, np.nan), A.astype(np.float32)], axis=-1).astype(np.float32)

    def fit(self, S: np.ndarray, A: np.ndarray, y: np.ndarray) -> "GBMStatic":
        self.model.fit(self.features(S, A), y)
        return self

    def score(self, S: np.ndarray, A: np.ndarray) -> np.ndarray:
        X = self.features(S, A)
        shape = X.shape[:-1]
        return self.model.decision_function(X.reshape(-1, X.shape[-1])).reshape(shape).astype(np.float32)
