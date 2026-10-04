"""Tune the static neural model on the validation fold only.

Trains on relations before 2014 and reports full- and warm-gallery MAP for
relations established in 2014-2017, next to v2's logistic recipe retrained on
the same pairs.  The test fold (2018 onwards) is never touched.

Usage:  python v3/tune_static.py default epochs=40 dim=96,z_dim=192 ...
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from baselines import fit_nonnegative_logistic
from common import configure_stdout
from features import StaticSimilarity
from metrics import query_metrics
from neural import DEFAULT_CONFIG, StaticEnsemble, build_pools, train_static
from world import load_world, make_task

CUTOFF, UNTIL = pd.Timestamp("2014-01-01"), pd.Timestamp("2018-01-01")


def evaluate(world, sim, task, scorers: dict, chunk: int = 64) -> dict[str, dict[str, float]]:
    queries = task.queries(np.arange(world.n), world)
    warm = task.snapshot.warm
    results: dict[str, dict[str, list]] = {name: {"full": [], "warm": []} for name in scorers}
    rng = np.random.default_rng(0)
    for start in range(0, len(queries), chunk):
        rows = queries[start : start + chunk]
        S, A = sim.block(rows)
        scores = {name: f(rows, S, A) for name, f in scorers.items()}
        excluded = task.excluded(rows, world).toarray()
        positive = (task.positives[rows].toarray() > 0) & ~excluded
        for i in range(len(rows)):
            for gallery, members in (("full", np.ones(world.n, bool)), ("warm", warm)):
                candidates = members & ~excluded[i]
                pos = positive[i] & candidates
                if pos.any() and (candidates & ~pos).any():
                    for name in scorers:
                        results[name][gallery].append(query_metrics(scores[name][i, candidates], pos[candidates], rng)[[0, 1]])
    return {
        name: {g: {"auc": float(np.mean([v[0] for v in r])), "map": float(np.mean([v[1] for v in r]))} for g, r in by.items()}
        for name, by in results.items()
    }


def parse_overrides(text: str) -> dict:
    if text in ("", "default"):
        return {}
    return {key: json.loads(value) for key, value in (item.split("=", 1) for item in text.split(","))}


def main() -> int:
    configure_stdout()
    world = load_world()
    sim = StaticSimilarity.from_world(world)
    task = make_task(world, "validation", CUTOFF, UNTIL)
    configs = [parse_overrides(a) for a in sys.argv[1:]] or [{}]
    pools = build_pools(world, sim, task.snapshot, DEFAULT_CONFIG, np.random.default_rng(1000))
    pool = pools["therapeutic"]
    logistic = fit_nonnegative_logistic(pool.S, pool.A, pool.y)
    scorers = {"v2_retrained": lambda rows, S, A: logistic.score(S, A)}
    for k, overrides in enumerate(configs):
        config = {**DEFAULT_CONFIG, **overrides}
        start = time.time()
        p = build_pools(world, sim, task.snapshot, config, np.random.default_rng(1000))
        model = train_static(world, sim, p, seed=0, config=config)
        ensemble = StaticEnsemble([model])
        scorers[f"neural {json.dumps(overrides)}"] = lambda rows, S, A, e=ensemble: e.block(rows, S, A)
        print(f"trained {overrides} in {time.time() - start:.0f}s, best epoch {max(model.history, key=lambda h: h['holdout_auc'])['epoch']}")
    for name, result in evaluate(world, sim, task, scorers).items():
        print(f"{name:60s} full MAP {result['full']['map']:.4f} AUC {result['full']['auc']:.4f} | warm MAP {result['warm']['map']:.4f} AUC {result['warm']['auc']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
