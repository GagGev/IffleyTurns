"""Shared paths, model registry and retrieval metrics for the global evaluation.

Nothing here imports a model version.  Each version is scored in its own process (their module names collide)
by the scripts in ``scorers/``; the benchmarks only read the score matrices those scripts write.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / ".data" / "global_eval"          # score matrices and task files (large, not committed)
RESULTS = Path(__file__).resolve().parent / "results"

# model id -> (version, label)
MODELS = {
    "v1": ("v1", "v1 weighted Jaccard"),
    "v1_drugfree": ("v1", "v1 weighted Jaccard (no approved drugs)"),
    "v2": ("v2", "v2 fusion, all 12 modalities"),
    "v2_drugfree": ("v2", "v2 fusion, drug-free view"),
    "v3_static": ("v3", "v3 static similarity (neural + fusion)"),
    "v3_forecast": ("v3", "v3 forecast (stacker)"),
    "v4_embedding": ("v4", "v4 multimodal embedding cosine"),
}
DRUG_DERIVED = ("drug", "drug_target", "ot_gene")   # modalities v3 does not have


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, document) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1, ensure_ascii=False, default=float), encoding="utf-8")


# --------------------------------------------------------------------------- metrics

_DISCOUNTS = 1.0 / np.log2(np.arange(2, 12))
RELATION_METRICS = ("map", "mrr", "hits@10", "precision@10", "ndcg@10", "auroc")


def relation_metrics(scores: np.ndarray, positives: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Retrieval metrics for one query: ``scores`` over candidates, boolean ``positives``.

    Ties are broken at random (seeded), so a constant score ranks like a random one; AUROC uses average ranks.
    """
    n, n_pos = scores.size, int(positives.sum())
    n_neg = n - n_pos
    scores = np.asarray(scores, dtype=np.float64)
    order = np.lexsort((rng.random(n), -scores))
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(1, n + 1)
    pr = np.sort(ranks[positives])
    ap = float(np.mean(np.arange(1, n_pos + 1) / pr))
    top = pr[pr <= 10]
    dcg = float(_DISCOUNTS[top - 1].sum())
    ideal = float(_DISCOUNTS[: min(n_pos, 10)].sum())
    auc = (rankdata(scores)[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return np.array([ap, 1.0 / pr[0], float(pr[0] <= 10), len(top) / 10, dcg / ideal, auc])


def bootstrap(values: np.ndarray, reps: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95% percentile interval of the mean over a resample of the rows."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    means = values[rng.integers(0, len(values), (reps, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bootstrap_stat(f, n: int, reps: int = 400, seed: int = 0) -> tuple[float, float, float]:
    """Value of f(index array) on all rows and a 95% interval over resamples of the rows."""
    rng = np.random.default_rng(seed)
    vals = [f(rng.integers(0, n, n)) for _ in range(reps)]
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(f(np.arange(n))), float(lo), float(hi)


def paired_difference(a: np.ndarray, b: np.ndarray, reps: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    return bootstrap(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64), reps, seed)


def rank_of_partner(scores: np.ndarray, query_col: int, partner_col: int) -> float:
    """1-based rank of the partner among all other diseases (ties share the average rank)."""
    s = scores.astype(np.float64).copy()
    s[query_col] = -np.inf
    rest = np.delete(s, query_col)
    target = s[partner_col]
    return 1 + float((rest > target).sum()) + 0.5 * float((rest == target).sum() - 1)


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)
