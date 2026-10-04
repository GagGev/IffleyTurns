"""Retrieval metrics for "given a disease, rank all other diseases" plus
bootstrap uncertainty.

Ties are broken uniformly at random (seeded) for rank-based metrics, and
handled by average ranks for AUC, so constant scores score exactly like a
random ranking instead of benefiting from the input order.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.stats import rankdata, wilcoxon

METRICS = ("auc", "map", "mrr", "hits@10", "precision@10", "recall@50", "ndcg@10")
PRIMARY_METRIC = "map"
_DISCOUNTS = 1.0 / np.log2(np.arange(2, 12))


def query_metrics(scores: np.ndarray, positives: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Metrics for one query; ``positives`` is a boolean mask over candidates."""

    n = scores.size
    n_pos = int(positives.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("A query needs at least one positive and one negative candidate.")
    scores = np.asarray(scores, dtype=np.float64)
    order = np.lexsort((rng.random(n), -scores))
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(1, n + 1)
    positive_ranks = np.sort(ranks[positives])

    average_ranks = rankdata(scores)
    auc = (average_ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    ap = float(np.mean(np.arange(1, n_pos + 1) / positive_ranks))
    top10 = positive_ranks[positive_ranks <= 10]
    dcg = float(_DISCOUNTS[top10 - 1].sum())
    idcg = float(_DISCOUNTS[: min(n_pos, 10)].sum())
    return np.array(
        [
            auc,
            ap,
            1.0 / positive_ranks[0],
            float(positive_ranks[0] <= 10),
            top10.size / 10.0,
            float((positive_ranks <= 50).sum()) / n_pos,
            dcg / idcg,
        ]
    )


def bootstrap_mean(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return (math.nan, math.nan, math.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_difference(
    a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> dict[str, float]:
    """Mean of a - b with a paired bootstrap 95% CI and one-sided p-values."""

    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    mean, low, high = bootstrap_mean(diff, n_boot, seed)
    try:
        p_wilcoxon = float(wilcoxon(diff, alternative="greater", zero_method="zsplit").pvalue) if np.any(diff) else 1.0
    except ValueError:
        p_wilcoxon = math.nan
    return {"mean_difference": mean, "ci_low": low, "ci_high": high, "p_wilcoxon_greater": p_wilcoxon}


def summarize(metric_rows: np.ndarray, seed: int = 0) -> dict[str, dict[str, float]]:
    summary = {}
    for j, metric in enumerate(METRICS):
        mean, low, high = bootstrap_mean(metric_rows[:, j], seed=seed + j)
        summary[metric] = {"mean": mean, "ci_low": low, "ci_high": high}
    return summary


def pooled_auc(positive_scores: Sequence[float], control_scores: Sequence[float]) -> float:
    pos = np.asarray(positive_scores, dtype=np.float64)
    neg = np.asarray(control_scores, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return math.nan
    ranks = rankdata(np.r_[pos, neg])
    return float((ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))
