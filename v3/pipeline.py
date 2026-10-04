"""Models trained at a date, the rolling-origin stacker, and prospective evaluation.

Everything used to predict relations after a date ``T`` is computed from
knowledge strictly before ``T``:

* the static neural ensemble and the static baselines are trained on the
  relations established before ``T``;
* the stacker is trained on earlier origins ``t0 < T``: features from the
  snapshot at ``t0`` (and a static ensemble trained before ``t0``), labels =
  relations established in ``[t0, T)``.  This mirrors the test exactly
  (features at ``T``, labels after ``T``), so graph and drug-history features
  are never evaluated on the relations that define them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from baselines import GBMStatic, LinearFusion, V1Baseline, fit_nonnegative_logistic, v2_shipped
from common import MODEL_DIR, timed
from features import ALL_FEATURES, FEATURE_GROUPS, StaticSimilarity, TemporalFeatures, stack
from metrics import METRICS, query_metrics
from modalities import MODALITIES
from neural import DEFAULT_CONFIG, StaticEnsemble, build_pools, train_ensemble
from world import World, make_task, pair_matrix

ORIGIN_OFFSETS = (2, 4, 6, 8)
MIN_ORIGIN_RELATIONS = 150
CHUNK = 64
NEGATIVES_UNIFORM = 60
NEGATIVES_WARM = 60
STACKER_PARAMS = {
    "max_iter": 600,
    "learning_rate": 0.04,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 50,
    "l2_regularization": 1.0,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 40,
}


def _without(*groups: str) -> tuple[str, ...]:
    removed = {f for g in groups for f in FEATURE_GROUPS[g]}
    return tuple(f for f in ALL_FEATURES if f not in removed)


STACKER_VARIANTS = {
    "v3_stacker": ALL_FEATURES,
    "stacker_no_static": _without("static"),
    "stacker_no_graph": _without("graph"),
    "stacker_no_history": _without("history"),
    "stacker_no_mechanism": _without("mechanism"),
    "stacker_static_only": tuple(f for f in ALL_FEATURES if f in FEATURE_GROUPS["static"] + FEATURE_GROUPS["node"]),
}


def _label(time: Optional[pd.Timestamp]) -> str:
    return "all" if time is None else time.strftime("%Y-%m-%d")


def _config_hash(world: World, config: dict[str, Any], time: Optional[pd.Timestamp]) -> str:
    known = len(world.snapshot(time).relations)
    payload = json.dumps({"config": config, "known": known, "nodes": world.n}, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:10]


@dataclass
class HybridStatic:
    """The v3 static (drug-free) relatedness score: the standardized sum of
    the neural ensemble and v2's non-negative logistic fusion of the raw
    cosines, both trained on relations established before one date.  On the
    validation fold the sum ranks better (AUC) than either part alone."""

    neural: StaticEnsemble
    linear: LinearFusion
    neural_scale: float
    linear_scale: float

    def combine(self, neural: np.ndarray, S: np.ndarray, A: np.ndarray) -> dict[str, np.ndarray]:
        linear = self.linear.score(S, A)
        return {
            "static_logit": (neural / self.neural_scale + linear / self.linear_scale).astype(np.float32),
            "neural_logit": neural.astype(np.float32),
            "linear_logit": linear,
        }

    def parts(self, rows: np.ndarray, S: np.ndarray, A: np.ndarray) -> dict[str, np.ndarray]:
        return self.combine(self.neural.block(rows, S, A), S, A)


def fit_hybrid(world: World, sim: StaticSimilarity, snapshot, config: dict[str, Any]) -> HybridStatic:
    neural = train_ensemble(world, sim, snapshot, config)
    pool = build_pools(world, sim, snapshot, config, np.random.default_rng(7))["therapeutic"]
    linear = fit_nonnegative_logistic(pool.S, pool.A, pool.y)
    sample = np.random.default_rng(0).choice(len(pool.y), min(20_000, len(pool.y)), replace=False)
    neural_scores = np.mean(
        [m.pair_logits(m.z[pool.a[sample]], m.z[pool.b[sample]], pool.S[sample], pool.A[sample]) for m in neural.models],
        axis=0,
    )
    return HybridStatic(
        neural=neural,
        linear=linear,
        neural_scale=float(np.std(neural_scores)) or 1.0,
        linear_scale=float(np.std(linear.score(pool.S[sample], pool.A[sample]))) or 1.0,
    )


@dataclass
class TimeModels:
    """Everything trained on relations established before ``time``."""

    time: Optional[pd.Timestamp]
    static: HybridStatic
    gbm_static: Optional[GBMStatic] = None


def time_models(
    world: World,
    sim: StaticSimilarity,
    time: Optional[pd.Timestamp],
    baselines: bool = False,
    config: Optional[dict[str, Any]] = None,
    cache: bool = True,
) -> TimeModels:
    config = {**DEFAULT_CONFIG, **(config or {})}
    path = MODEL_DIR / "time_models" / f"{_label(time)}_{_config_hash(world, config, time)}.joblib"
    snapshot = world.snapshot(time)
    changed = False
    if cache and path.is_file():
        models = joblib.load(path)
    else:
        with timed(f"Training the static models on {len(snapshot.relations)} relations before {_label(time)}"):
            models = TimeModels(time=time, static=fit_hybrid(world, sim, snapshot, config))
        changed = True
    if baselines and models.gbm_static is None:
        pool = build_pools(world, sim, snapshot, config, np.random.default_rng(7))["therapeutic"]
        models.gbm_static = GBMStatic().fit(pool.S, pool.A, pool.y)
        changed = True
    if cache and changed:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(models, path, compress=3)
    return models


def origins_for(world: World, cutoff: Optional[pd.Timestamp], end_of_data: pd.Timestamp) -> list[pd.Timestamp]:
    anchor = end_of_data if cutoff is None else cutoff
    out = []
    for years in ORIGIN_OFFSETS:
        origin = anchor - pd.DateOffset(years=years)
        if len(world.snapshot(origin).relations) >= MIN_ORIGIN_RELATIONS:
            out.append(origin)
    return sorted(out)


def _chunks(rows: np.ndarray, size: int = CHUNK) -> Iterable[np.ndarray]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


# --------------------------------------------------------------------------- stacker


@dataclass
class Stacker:
    name: str
    features: tuple[str, ...]
    model: HistGradientBoostingClassifier

    def score(self, block: dict[str, np.ndarray]) -> np.ndarray:
        X = stack(block, self.features)
        shape = X.shape[:-1]
        return self.model.decision_function(X.reshape(-1, X.shape[-1])).reshape(shape).astype(np.float32)


@dataclass
class StackerData:
    """Sampled candidate pairs.  Negatives are drawn separately among warm
    (already designated) and cold candidates and weighted by the inverse of
    their sampling rate, so the weighted sample has the true candidate mix."""

    X: np.ndarray
    y: np.ndarray
    weight: np.ndarray
    origin: np.ndarray
    query: np.ndarray


def collect_stacker_data(
    world: World,
    sim: StaticSimilarity,
    cutoff: Optional[pd.Timestamp],
    origins: Sequence[pd.Timestamp],
    config: Optional[dict[str, Any]] = None,
    seed: int = 0,
) -> StackerData:
    rng = np.random.default_rng(seed)
    parts: list[tuple[np.ndarray, ...]] = []
    for origin in origins:
        models = time_models(world, sim, origin, config=config)
        task = make_task(world, f"origin {_label(origin)}", origin, cutoff)
        features = TemporalFeatures(world, task.snapshot, sim, models.static.parts)
        queries = task.queries(np.arange(world.n), world)
        warm = task.snapshot.warm
        with timed(f"Stacker rows from origin {_label(origin)}: {len(task.relations)} new relations, {len(queries)} queries"):
            for rows in _chunks(queries):
                X = stack(features.block(rows), ALL_FEATURES)
                excluded = task.excluded(rows, world).toarray()
                positive = (task.positives[rows].toarray() > 0) & ~excluded
                for i, query in enumerate(rows):
                    pos = np.flatnonzero(positive[i])
                    negatives = np.flatnonzero(~excluded[i] & ~positive[i])
                    cols, weights = [pos], [np.ones(pos.size)]
                    for stratum, k in ((negatives[warm[negatives]], NEGATIVES_WARM), (negatives[~warm[negatives]], NEGATIVES_UNIFORM)):
                        if stratum.size:
                            take = rng.choice(stratum, min(k, stratum.size), replace=False)
                            cols.append(take)
                            weights.append(np.full(take.size, stratum.size / take.size))
                    cols = np.concatenate(cols)
                    parts.append(
                        (
                            X[i, cols],
                            np.r_[np.ones(pos.size), np.zeros(cols.size - pos.size)],
                            np.concatenate(weights),
                            np.full(cols.size, origin.year),
                            np.full(cols.size, query),
                        )
                    )
    X, y, w, o, q = (np.concatenate(x) for x in zip(*parts))
    # One global factor balances the classes without changing the warm/cold
    # odds ratio that the inverse-probability weights restore.
    negative = y == 0
    w[negative] *= 10.0 * y.sum() / w[negative].sum()
    return StackerData(X=X.astype(np.float32), y=y, weight=w, origin=o, query=q)


def fit_stackers(data: StackerData, variants: dict[str, tuple[str, ...]] = STACKER_VARIANTS, seed: int = 0) -> dict[str, Stacker]:
    column = {f: j for j, f in enumerate(ALL_FEATURES)}
    stackers = {}
    for name, names in variants.items():
        model = HistGradientBoostingClassifier(**STACKER_PARAMS, random_state=seed)
        model.fit(data.X[:, [column[f] for f in names]], data.y, sample_weight=data.weight)
        stackers[name] = Stacker(name=name, features=tuple(names), model=model)
    return stackers


def stacker_importance(stacker: Stacker, data: StackerData, seed: int = 0, n_rows: int = 40_000) -> dict[str, float]:
    """Permutation importance (drop in pooled AUC) of each feature family."""

    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    column = {f: j for j, f in enumerate(ALL_FEATURES)}
    take = rng.choice(len(data.y), min(n_rows, len(data.y)), replace=False)
    X = data.X[take][:, [column[f] for f in stacker.features]]
    y, w = data.y[take], data.weight[take]
    base = roc_auc_score(y, stacker.model.decision_function(X), sample_weight=w)
    out = {}
    for group, names in FEATURE_GROUPS.items():
        idx = [stacker.features.index(f) for f in names if f in stacker.features]
        if not idx:
            continue
        shuffled = X.copy()
        perm = rng.permutation(len(y))
        shuffled[:, idx] = shuffled[perm][:, idx]
        out[group] = float(base - roc_auc_score(y, stacker.model.decision_function(shuffled), sample_weight=w))
    return out


# --------------------------------------------------------------------------- evaluation

Scorer = Callable[[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray], np.ndarray]


@dataclass
class FoldModels:
    """Every model scored in a fold, as functions of (rows, feature block, S, A)."""

    scorers: dict[str, Scorer]
    stackers: dict[str, Stacker]
    stacker_data: StackerData
    time_models: TimeModels
    features: TemporalFeatures


def build_fold_models(
    world: World,
    sim: StaticSimilarity,
    cutoff: Optional[pd.Timestamp],
    end_of_data: pd.Timestamp,
    config: Optional[dict[str, Any]] = None,
    seed: int = 0,
) -> FoldModels:
    origins = origins_for(world, cutoff, end_of_data)
    data = collect_stacker_data(world, sim, cutoff, origins, config, seed)
    with timed(f"Fitting {len(STACKER_VARIANTS)} stackers on {len(data.y)} rows ({int(data.y.sum())} positive)"):
        stackers = fit_stackers(data, seed=seed)
    models = time_models(world, sim, cutoff, baselines=True, config=config)
    features = TemporalFeatures(world, world.snapshot(cutoff), sim, models.static.parts)
    v1 = V1Baseline(world.bundle.v1_sets, world.ids)
    shipped = v2_shipped()
    rng = np.random.default_rng(seed)

    scorers: dict[str, Scorer] = {
        "random": lambda rows, block, S, A: rng.random((len(rows), world.n)).astype(np.float32),
        "degree": lambda rows, block, S, A: np.broadcast_to(
            features.nodes.log_designations[None, :] + 1e-3 * np.log1p(features.nodes.degree)[None, :], (len(rows), world.n)
        ),
        "adamic_adar": lambda rows, block, S, A: block["adamic_adar"],
        "drug_mechanism": lambda rows, block, S, A: block["target_cosine"]
        + block["gene_target"]
        + 0.5 * block["target_pathway_cosine"],
        "v1_weighted_jaccard": lambda rows, block, S, A: v1.block(rows),
    }
    if shipped is not None:
        scorers["v2_shipped"] = lambda rows, block, S, A: shipped.score(S, A)
    scorers["v2_retrained"] = lambda rows, block, S, A: block["linear_logit"]
    scorers["gbm_static"] = lambda rows, block, S, A: models.gbm_static.score(S, A)
    scorers["v3_neural"] = lambda rows, block, S, A: block["neural_logit"]
    scorers["v3_static"] = lambda rows, block, S, A: block["static_logit"]
    for name, stacker in stackers.items():
        scorers[name] = lambda rows, block, S, A, s=stacker: s.score(block)
    return FoldModels(scorers=scorers, stackers=stackers, stacker_data=data, time_models=models, features=features)


@dataclass
class FoldResult:
    name: str
    cutoff: str
    per_query: pd.DataFrame
    global_ranking: dict[str, dict[str, float]]
    importance: dict[str, float]
    info: dict[str, Any] = field(default_factory=dict)


def evaluate_fold(
    world: World,
    sim: StaticSimilarity,
    name: str,
    cutoff: pd.Timestamp,
    until: Optional[pd.Timestamp],
    end_of_data: pd.Timestamp,
    config: Optional[dict[str, Any]] = None,
    seed: int = 0,
    top_k: Sequence[int] = (25, 100, 500),
) -> FoldResult:
    fold = build_fold_models(world, sim, cutoff, end_of_data, config, seed)
    task = make_task(world, name, cutoff, until)
    approved = task.relations[task.relations["approved_both"]]
    approved_matrix = pair_matrix(
        world.n, [world.index[a] for a in approved["a"]], [world.index[b] for b in approved["b"]]
    )
    snapshot = task.snapshot
    warm = snapshot.warm
    galleries = {"full": np.ones(world.n, dtype=bool), "warm": warm}
    queries = task.queries(np.arange(world.n), world)
    query_set = set(queries.tolist())
    rows_all = np.union1d(queries, np.flatnonzero(warm))
    model_names = list(fold.scorers)
    rngs = {m: np.random.default_rng(seed + k) for k, m in enumerate(model_names)}
    records: list[dict[str, Any]] = []
    pair_scores: dict[str, list[np.ndarray]] = {m: [] for m in model_names}
    pair_labels: list[np.ndarray] = []

    with timed(f"Scoring {len(model_names)} models: {len(queries)} queries, {int(warm.sum())} warm nodes after {_label(cutoff)}"):
        for rows in _chunks(rows_all):
            block = fold.features.block(rows)
            S = np.nan_to_num(np.stack([block[f"sim_{m}"] for m in MODALITIES], axis=-1))
            A = ~np.isnan(np.stack([block[f"sim_{m}"] for m in MODALITIES], axis=-1))
            scores = {m: np.asarray(fold.scorers[m](rows, block, S, A), dtype=np.float32) for m in model_names}
            excluded = task.excluded(rows, world).toarray()
            positive = (task.positives[rows].toarray() > 0) & ~excluded
            approved_rows = approved_matrix[rows].toarray() > 0
            for i, query in enumerate(rows):
                if warm[query]:
                    later = np.flatnonzero(warm & ~excluded[i] & (np.arange(world.n) > query))
                    if later.size:
                        pair_labels.append(positive[i, later])
                        for m in model_names:
                            pair_scores[m].append(scores[m][i, later])
                if query not in query_set:
                    continue
                approved_mask = approved_rows[i]
                for gallery, members in galleries.items():
                    for subset in ("all", "approved"):
                        if subset == "approved" and gallery != "full":
                            continue
                        candidates = members & ~excluded[i]
                        pos = positive[i] & candidates
                        if subset == "approved":
                            keep = approved_mask & pos
                            candidates = candidates & ~(pos & ~keep)
                            pos = keep
                        if not pos.any() or not (candidates & ~pos).any():
                            continue
                        base = {
                            "fold": name,
                            "gallery": gallery if subset == "all" else "full_approved",
                            "query": world.ids[query],
                            "query_warm": bool(warm[query]),
                            "query_group": bool(world.is_group[query]),
                            "query_oncology": bool(world.oncology[query]),
                            "n_positives": int(pos.sum()),
                            "n_candidates": int(candidates.sum()),
                        }
                        for m in model_names:
                            values = query_metrics(scores[m][i, candidates], pos[candidates], rngs[m])
                            records.append({**base, "model": m, **dict(zip(METRICS, values))})

    labels = np.concatenate(pair_labels) if pair_labels else np.zeros(0, dtype=bool)
    global_ranking: dict[str, dict[str, float]] = {}
    tie_break = np.random.default_rng(seed).random(labels.size)
    for m in model_names:
        values = np.concatenate(pair_scores[m]) if pair_scores[m] else np.zeros(0)
        order = np.lexsort((tie_break, -values))
        global_ranking[m] = {f"precision@{k}": float(labels[order[:k]].mean()) for k in top_k if labels.size >= k}
    global_ranking["base_rate"] = {"positives": int(labels.sum()), "pairs": int(labels.size), "rate": float(labels.mean()) if labels.size else 0.0}
    importance = stacker_importance(fold.stackers["v3_stacker"], fold.stacker_data, seed)
    info = {
        "cutoff": _label(cutoff),
        "until": _label(until) if until is not None else "end of data",
        "known_relations": int(len(snapshot.relations)),
        "new_relations": int(len(task.relations)),
        "queries_full": int(len(queries)),
        "warm_nodes": int(warm.sum()),
        "stacker_rows": int(len(fold.stacker_data.y)),
        "stacker_positives": int(fold.stacker_data.y.sum()),
        "stacker_origins": sorted({int(x) for x in fold.stacker_data.origin}),
        "static_training_history": [m.history for m in fold.time_models.static.neural.models],
        "v2_retrained_coefficients": fold.time_models.static.linear.coefficients(),
    }
    return FoldResult(
        name=name,
        cutoff=_label(cutoff),
        per_query=pd.DataFrame(records),
        global_ranking=global_ranking,
        importance=importance,
        info=info,
    )
