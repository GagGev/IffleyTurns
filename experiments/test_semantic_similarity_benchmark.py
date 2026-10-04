"""Focused tests for the isolated semantic-similarity experiment."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).with_name("02_semantic_similarity_benchmark.py")
SPEC = importlib.util.spec_from_file_location("semantic_similarity_benchmark", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCHMARK
SPEC.loader.exec_module(BENCHMARK)


def test_set_statistics_distinguishes_overlap_and_balance() -> None:
    values = BENCHMARK._set_statistics({"a", "b"}, {"b", "c", "d"})

    assert values == pytest.approx(
        [
            1.0,
            1.0 / 4.0,
            1.0 / 2.0,
            2.0 / 3.0,
            math.log1p(1),
            math.log1p(4),
        ]
    )


def test_set_statistics_marks_missing_pair_unavailable() -> None:
    assert BENCHMARK._set_statistics(set(), {"a"}) == [0.0] * len(
        BENCHMARK.SET_STAT_SUFFIXES
    )


def test_weighted_profile_statistics_are_bounded() -> None:
    weighted_jaccard, cosine = BENCHMARK._weighted_profile_statistics(
        {"root": 0.1, "specific": 2.0},
        {"root": 0.1, "other": 1.5},
    )

    assert 0.0 < weighted_jaccard < 1.0
    assert 0.0 < cosine < 1.0


def test_feature_names_are_unique_and_compact() -> None:
    names = BENCHMARK.improved_feature_names()

    assert len(names) == 95
    assert len(names) == len(set(names))
    assert len(names) < len(BENCHMARK.CURRENT_FEATURE_NAMES)


def test_average_descending_rank_handles_ties() -> None:
    scores = np.asarray([0.7, 0.9, 0.7, 0.1])

    assert BENCHMARK._average_descending_rank(scores, 0) == pytest.approx(2.5)


def test_ranking_bootstrap_reports_positive_improvement() -> None:
    result = BENCHMARK._paired_ranking_bootstrap(
        [4.0, 3.0, 2.0],
        [2.0, 1.0, 1.0],
        repetitions=200,
    )

    assert result["mrr_delta"] > 0
    assert result["mrr_probability_improved"] == 1.0


def test_output_guard_rejects_paths_outside_experiments(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be inside"):
        BENCHMARK._assert_output_is_isolated(tmp_path)
