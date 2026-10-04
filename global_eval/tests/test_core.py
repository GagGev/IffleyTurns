import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import core
import run as orchestrator


def test_perfect_ranking_scores_one():
    scores = np.array([5.0, 4.0, 1.0, 0.5, 0.1])
    positives = np.array([True, True, False, False, False])
    ap, mrr, hits, p10, ndcg, auc = core.relation_metrics(scores, positives, np.random.default_rng(0))
    assert (ap, mrr, hits, auc) == (1.0, 1.0, 1.0, 1.0) and abs(ndcg - 1.0) < 1e-9 and p10 == 0.2


def test_constant_scores_rank_like_random():
    n = 2000
    positives = np.zeros(n, bool); positives[:20] = True
    values = [core.relation_metrics(np.zeros(n), positives, np.random.default_rng(s))[5] for s in range(5)]
    assert all(abs(v - 0.5) < 1e-9 for v in values)


def test_rank_of_partner_ties_share_average_rank():
    scores = np.array([9.0, 1.0, 1.0, 1.0, 0.0])      # query is column 0; partner ties with two others
    assert core.rank_of_partner(scores, 0, 1) == 2.0


def test_bootstrap_interval_contains_mean():
    values = np.random.default_rng(1).normal(size=200)
    mean, lo, hi = core.bootstrap(values, reps=500)
    assert lo < mean < hi


def test_v2_v4_selection_never_schedules_v3(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "DATA", tmp_path)
    launched = []

    def capture(command, check):
        launched.append(Path(command[-1]).name)

    monkeypatch.setattr(orchestrator.subprocess, "run", capture)
    orchestrator.score(False, ["v2", "v2_drugfree", "v4_embedding"])
    assert launched == ["v2_side.py", "v4_side.py"]
    assert "v3_side.py" not in launched
