"""Similarity computation and the disease-pair scoring models.

All models consume the same per-modality cosine similarities ``S`` and
availability flags ``A`` (both diseases annotated in that modality), shaped
``(queries, gallery, modalities)``.  Masked modalities are zero/unavailable,
which is how benchmarks hide the modalities that define their labels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Optional, Sequence

import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from benchmarks import Relation, sample_training_pairs


class Gallery:
    """Pre-transposed gallery matrices for fast query-vs-gallery similarity."""

    def __init__(self, engine: "SimilarityEngine", indices: Sequence[int]):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.ids = [engine.ids[i] for i in self.indices]
        self.transposed = {m: engine.matrices[m][self.indices].T.tocsr() for m in engine.modalities}
        self.available = engine.available[self.indices]


class SimilarityEngine:
    def __init__(self, matrices: dict[str, sp.csr_matrix], ids: Sequence[str]):
        self.modalities = tuple(matrices)
        self.matrices = {m: x.tocsr() for m, x in matrices.items()}
        self.ids = list(ids)
        self.index = {d: i for i, d in enumerate(self.ids)}
        self.available = np.stack([self.matrices[m].getnnz(axis=1) > 0 for m in self.modalities], axis=1)

    def mask_vector(self, masked: Sequence[str]) -> np.ndarray:
        return np.array([m in masked for m in self.modalities], dtype=bool)

    def gallery(self, ids: Sequence[str]) -> Gallery:
        return Gallery(self, [self.index[d] for d in ids])

    def rows(self, ids: Sequence[str]) -> tuple[dict[str, sp.csr_matrix], np.ndarray]:
        idx = np.array([self.index[d] for d in ids], dtype=np.int64)
        return {m: self.matrices[m][idx] for m in self.modalities}, self.available[idx]

    def pair_features(
        self, a_ids: Sequence[str], b_ids: Sequence[str], masked: Sequence[str] = ()
    ) -> tuple[np.ndarray, np.ndarray]:
        a = np.array([self.index[d] for d in a_ids], dtype=np.int64)
        b = np.array([self.index[d] for d in b_ids], dtype=np.int64)
        mask = self.mask_vector(masked)
        S = np.zeros((len(a), len(self.modalities)), dtype=np.float32)
        for j, m in enumerate(self.modalities):
            if mask[j]:
                continue
            X = self.matrices[m]
            S[:, j] = np.asarray(X[a].multiply(X[b]).sum(axis=1)).ravel()
        A = self.available[a] & self.available[b]
        A[:, mask] = False
        S[~A] = 0.0
        return S, A

    def block(
        self,
        query_matrices: dict[str, sp.csr_matrix],
        query_available: np.ndarray,
        gallery: Gallery,
        masked: Sequence[str] = (),
    ) -> tuple[np.ndarray, np.ndarray]:
        mask = self.mask_vector(masked)
        n_q = query_available.shape[0]
        S = np.zeros((n_q, len(gallery.indices), len(self.modalities)), dtype=np.float32)
        for j, m in enumerate(self.modalities):
            if mask[j]:
                continue
            product = query_matrices[m] @ gallery.transposed[m]
            S[:, :, j] = product.toarray() if sp.issparse(product) else product
        A = query_available[:, None, :] & gallery.available[None, :, :]
        A[:, :, mask] = False
        S[~A] = 0.0
        return S, A


# --------------------------------------------------------------------------- models


class RandomModel:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def score(self, S: np.ndarray, A: np.ndarray) -> np.ndarray:
        return self.rng.random(S.shape[:2])


class SingleModality:
    def __init__(self, modality: str, modalities: Sequence[str]):
        self.name = f"only_{modality}"
        self.column = list(modalities).index(modality)

    def score(self, S, A):
        return S[:, :, self.column]


class UniformFusion:
    """Mean of the available modality similarities, shrunk toward 0 when few are available."""

    name = "uniform_mean"

    def __init__(self, shrinkage: float = 1.0):
        self.shrinkage = shrinkage

    def score(self, S, A):
        denominator = self.shrinkage + A.sum(axis=2)
        return np.divide(S.sum(axis=2), denominator, out=np.zeros(S.shape[:2], dtype=np.float32), where=denominator > 0)


@dataclass
class LinearFusion:
    """Logistic-regression fusion: logit = b + sum_m (w_m * sim_m + v_m * available_m)."""

    modalities: tuple[str, ...]
    similarity_weights: np.ndarray
    availability_weights: np.ndarray
    intercept: float
    name: str = "logistic_fusion"
    params: dict[str, Any] = field(default_factory=dict)

    def score(self, S, A):
        return (
            np.tensordot(S, self.similarity_weights, axes=([2], [0]))
            + np.tensordot(A.astype(np.float32), self.availability_weights, axes=([2], [0]))
            + self.intercept
        )

    def similarity_contributions(self, S: np.ndarray) -> np.ndarray:
        """Per-modality logit contributions of the similarities (last axis = modality)."""

        return S * self.similarity_weights

    def availability_adjustment(self, A: np.ndarray) -> np.ndarray:
        """Logit shift from which modalities are annotated for both diseases."""

        return A.astype(np.float32) @ self.availability_weights

    def probability(self, logit: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-logit))

    def coefficients(self) -> list[dict[str, Any]]:
        return [
            {"modality": m, "similarity_weight": float(w), "availability_weight": float(v)}
            for m, w, v in zip(self.modalities, self.similarity_weights, self.availability_weights)
        ]


class GBMFusion:
    """Gradient-boosted trees over similarities (NaN when unavailable) and availability."""

    def __init__(self, model: HistGradientBoostingClassifier, name: str = "gbm_fusion"):
        self.model = model
        self.name = name

    @staticmethod
    def features(S: np.ndarray, A: np.ndarray) -> np.ndarray:
        values = np.where(A, S, np.nan).astype(np.float32)
        return np.concatenate([values, A.astype(np.float32)], axis=-1)

    def score(self, S, A):
        X = self.features(S, A)
        shape = X.shape[:-1]
        return self.model.decision_function(X.reshape(-1, X.shape[-1])).reshape(shape)


# --------------------------------------------------------------------------- v1 baseline

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


class V1Baseline:
    """The v1 metric (root ``evaluation.py``): weighted Jaccard over raw annotation
    sets plus exp(-|log10 prevalence difference|), renormalized over the
    families available for both diseases."""

    name = "v1_weighted_jaccard"

    def __init__(self, v1_sets: dict[str, dict[str, Any]], ids: Sequence[str]):
        self.ids = list(ids)
        self.index = {d: i for i, d in enumerate(self.ids)}
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
        self.sizes = {f: np.asarray(x.sum(axis=1)).ravel() for f, x in self.binary.items()}
        prevalence = [v1_sets[d]["prevalence"] for d in self.ids]
        self.log_prevalence = np.array([math.log10(p) if p else np.nan for p in prevalence])

    def block(self, query_ids: Sequence[str], gallery_ids: Sequence[str], masked: Sequence[str]) -> np.ndarray:
        q = np.array([self.index[d] for d in query_ids])
        g = np.array([self.index[d] for d in gallery_ids])
        numerator = np.zeros((len(q), len(g)))
        denominator = np.zeros((len(q), len(g)))
        for family, weight in V1_WEIGHTS.items():
            if family in masked:
                continue
            if family == "prevalence":
                lq, lg = self.log_prevalence[q][:, None], self.log_prevalence[g][None, :]
                available = ~np.isnan(lq) & ~np.isnan(lg)
                similarity = np.where(available, np.exp(-np.abs(np.nan_to_num(lq) - np.nan_to_num(lg))), 0.0)
            else:
                X = self.binary[family]
                intersection = (X[q] @ X[g].T).toarray()
                sq, sg = self.sizes[family][q][:, None], self.sizes[family][g][None, :]
                available = (sq > 0) & (sg > 0)
                union = np.maximum(sq + sg - intersection, 1.0)
                similarity = np.where(available, intersection / union, 0.0)
            numerator += weight * similarity
            denominator += weight * available
        return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


# --------------------------------------------------------------------------- training


@dataclass
class TrainingSet:
    S: np.ndarray
    A: np.ndarray
    y: np.ndarray
    weight: np.ndarray
    benchmark: np.ndarray


def build_training_set(
    engine: SimilarityEngine,
    relations: Sequence[Relation],
    members: set[str],
    negatives_per_positive: int = 10,
    seed: int = 0,
) -> TrainingSet:
    """Masked multi-task training pairs: each benchmark's pairs hide that
    benchmark's own label-source modalities, for positives and negatives alike."""

    rng = np.random.default_rng(seed)
    parts = []
    for b, relation in enumerate(relations):
        pairs, labels = sample_training_pairs(relation, members, negatives_per_positive, rng)
        if not pairs:
            continue
        S, A = engine.pair_features([p[0] for p in pairs], [p[1] for p in pairs], masked=relation.spec.masked)
        weight = np.full(len(pairs), 1.0 / len(pairs))
        parts.append((S, A, labels, weight, np.full(len(pairs), b)))
    S, A, y, w, bench = (np.concatenate(x) for x in zip(*parts))
    w = w * len(y) / w.sum()
    return TrainingSet(S=S, A=A, y=y, weight=w, benchmark=bench)


def fit_logistic(
    data: TrainingSet,
    modalities: Sequence[str],
    C: float = 1.0,
    use_availability: bool = True,
    nonnegative: bool = True,
    name: str = "logistic_fusion",
    init: Optional[LinearFusion] = None,
) -> LinearFusion:
    """Weighted L2 logistic regression (sklearn's C convention).

    With ``nonnegative`` the similarity weights are constrained to be >= 0, so
    more similarity in any modality can never lower a pair's score.  Without
    it, correlated modalities (gene, pathway, Open Targets genes) can receive
    offsetting negative weights that make explanations misleading.
    """

    X = np.hstack([data.S, data.A.astype(np.float32)]) if use_availability else data.S
    n = len(modalities)
    if not nonnegative:
        model = LogisticRegression(C=C, max_iter=5000)
        model.fit(X, data.y, sample_weight=data.weight)
        coef, intercept = model.coef_.ravel(), float(model.intercept_[0])
    else:
        start = None
        if init is not None:
            start = np.r_[init.similarity_weights, init.availability_weights if use_availability else [], init.intercept]
        coef, intercept = _bounded_logistic(X.astype(np.float64), data.y, data.weight, C, n_nonnegative=n, start=start)
    availability = coef[n:] if use_availability else np.zeros(n)
    return LinearFusion(
        modalities=tuple(modalities),
        similarity_weights=np.asarray(coef[:n], dtype=np.float32),
        availability_weights=np.asarray(availability, dtype=np.float32),
        intercept=intercept,
        name=name,
        params={"C": C, "use_availability": use_availability, "nonnegative": nonnegative},
    )


def _bounded_logistic(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, C: float, n_nonnegative: int, start: Optional[np.ndarray] = None
) -> tuple[np.ndarray, float]:
    """Minimize 0.5*||w||^2 + C * sum_i weight_i * logloss_i with w[:n_nonnegative] >= 0."""

    from scipy.optimize import minimize

    total = weight.sum()
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
    x0 = np.zeros(X.shape[1] + 1) if start is None else np.asarray(start, dtype=np.float64)
    result = minimize(objective, x0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 5000})
    return result.x[:-1], float(result.x[-1])


class PatternFusion:
    """Logistic fusion refitted to the modalities each query actually has.

    A linear fusion cannot express "rely on the gene when pathways are
    missing": correlated modalities share one fixed weight budget.  Refitting
    on the same training pairs with the query's missing modalities hidden
    (a pattern submodel) moves their weight onto the modalities it does have,
    e.g. a new disease with only a curated gene and no Open Targets profile.
    Submodels are cached by the set of hidden modalities.
    """

    def __init__(self, data: TrainingSet, base: LinearFusion, name: str = "pattern_fusion"):
        self.data = data
        self.base = base
        self.modalities = base.modalities
        self.name = name
        self.submodels: dict[frozenset[str], LinearFusion] = {frozenset(): base}

    def submodel(self, hidden: Sequence[str]) -> LinearFusion:
        key = frozenset(hidden)
        if key not in self.submodels:
            columns = [self.modalities.index(m) for m in key]
            S, A = self.data.S.copy(), self.data.A.copy()
            S[:, columns] = 0.0
            A[:, columns] = False
            start = replace(
                self.base,
                similarity_weights=self.base.similarity_weights.copy(),
                availability_weights=self.base.availability_weights.copy(),
            )
            start.similarity_weights[columns] = 0.0
            start.availability_weights[columns] = 0.0
            params = self.base.params
            self.submodels[key] = fit_logistic(
                TrainingSet(S=S, A=A, y=self.data.y, weight=self.data.weight, benchmark=self.data.benchmark),
                self.modalities,
                params["C"],
                params["use_availability"],
                params.get("nonnegative", True),
                name=f"{self.base.name} without {', '.join(sorted(key))}",
                init=start,
            )
        return self.submodels[key]

    def hidden_for(self, present: np.ndarray) -> list[str]:
        return [m for m, p in zip(self.modalities, present) if not p]

    def score(self, S: np.ndarray, A: np.ndarray) -> np.ndarray:
        present = A.any(axis=1)
        patterns, inverse = np.unique(present, axis=0, return_inverse=True)
        inverse = inverse.ravel()
        scores = np.empty(S.shape[:2], dtype=np.float32)
        for k, pattern in enumerate(patterns):
            rows = inverse == k
            scores[rows] = self.submodel(self.hidden_for(pattern)).score(S[rows], A[rows])
        return scores


def fit_gbm(data: TrainingSet, seed: int = 0, name: str = "gbm_fusion", **params: Any) -> GBMFusion:
    settings = dict(
        learning_rate=0.05,
        max_iter=500,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=seed,
    )
    settings.update(params)
    model = HistGradientBoostingClassifier(**settings)
    model.fit(GBMFusion.features(data.S, data.A), data.y, sample_weight=data.weight)
    return GBMFusion(model, name=name)


def subset(data: TrainingSet, benchmark_index: int) -> TrainingSet:
    keep = data.benchmark == benchmark_index
    weight = data.weight[keep]
    return TrainingSet(
        S=data.S[keep], A=data.A[keep], y=data.y[keep], weight=weight * keep.sum() / weight.sum(), benchmark=data.benchmark[keep]
    )
