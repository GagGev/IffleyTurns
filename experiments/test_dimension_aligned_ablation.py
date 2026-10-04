"""Focused tests for the dimension-aligned follow-up experiment."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("03_dimension_aligned_ablation.py")
SPEC = importlib.util.spec_from_file_location("dimension_aligned_ablation", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ABLATION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ABLATION
SPEC.loader.exec_module(ABLATION)


def test_each_primary_dimension_selects_compact_features() -> None:
    all_names = ABLATION.SEMANTIC.improved_feature_names()

    for dimension in ABLATION.PRIMARY_DIMENSIONS:
        indices = ABLATION.aligned_feature_indices(dimension)
        names = [all_names[index] for index in indices]
        assert names
        assert len(names) < len(all_names)
        assert not any(name.startswith("description_") for name in names)


def test_genetic_selection_uses_curated_gene_and_inheritance_features() -> None:
    names = ABLATION.SEMANTIC.improved_feature_names()
    selected = {
        names[index]
        for index in ABLATION.aligned_feature_indices("genetic")
    }

    assert "genes_trusted_jaccard" in selected
    assert "genes_high_confidence_ot_jaccard" in selected
    assert "inheritance_jaccard" in selected
    assert "phenotypes_ic_cosine" not in selected


def test_paper_tokens_parse_semicolon_ids() -> None:
    assert ABLATION._paper_tokens(
        {"paper_ids": "PMID:1;PMID:2"}
    ) == {"PMID:1", "PMID:2"}


def test_bootstrap_recognizes_uniformly_better_predictions() -> None:
    truth = np.asarray([0.2, 0.5, 0.8])
    current = np.asarray([0.5, 0.2, 0.5])
    aligned = np.asarray([0.25, 0.45, 0.75])
    weights = np.ones(3)

    result = ABLATION._paired_bootstrap(
        truth,
        current,
        aligned,
        weights,
        repetitions=200,
    )

    assert result["rmse_delta"] > 0
    assert result["rmse_probability_improved"] == 1.0
