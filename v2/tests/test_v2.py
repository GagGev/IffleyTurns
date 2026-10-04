"""Tests for the v2 similarity pipeline.

Unit tests use small synthetic inputs.  Integration tests use the cached
database bundle or the saved production model and are skipped when those
have not been built.

Run:  python -m pytest v2/tests -q
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

V2_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_DIR))

from common import FEATURE_DIR, assign_split, pair_key  # noqa: E402
from metrics import METRICS, query_metrics  # noqa: E402
from modalities import TextEncoder, WeightedItemEncoder  # noqa: E402
from scoring import (  # noqa: E402
    LinearFusion,
    PatternFusion,
    SimilarityEngine,
    TrainingSet,
    _bounded_logistic,
    fit_logistic,
)

HAS_FEATURES = (FEATURE_DIR / "diseases.parquet").is_file()


def metric(values: np.ndarray, name: str) -> float:
    return float(values[METRICS.index(name)])


# --------------------------------------------------------------------------- metrics


def test_query_metrics_match_hand_computed_values() -> None:
    scores = np.array([3.0, 2.0, 1.0, 0.0])
    positives = np.array([True, False, True, False])
    values = query_metrics(scores, positives, np.random.default_rng(0))

    assert metric(values, "auc") == pytest.approx(0.75)
    assert metric(values, "map") == pytest.approx((1 / 1 + 2 / 3) / 2)
    assert metric(values, "mrr") == pytest.approx(1.0)
    assert metric(values, "hits@10") == 1.0
    assert metric(values, "precision@10") == pytest.approx(0.2)
    assert metric(values, "recall@50") == 1.0
    assert metric(values, "ndcg@10") == pytest.approx((1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3)))


def test_constant_scores_rank_like_random_not_like_input_order() -> None:
    scores = np.zeros(20)
    positives = np.zeros(20, dtype=bool)
    positives[:2] = True
    runs = np.array([query_metrics(scores, positives, np.random.default_rng(seed)) for seed in range(500)])

    assert np.all(runs[:, METRICS.index("auc")] == 0.5)
    # Positives sit first in the input; order-based tie breaking would give MAP = 1.
    assert runs[:, METRICS.index("map")].mean() < 0.4


def test_query_metrics_reject_degenerate_queries() -> None:
    with pytest.raises(ValueError):
        query_metrics(np.arange(3.0), np.zeros(3, dtype=bool), np.random.default_rng(0))


# --------------------------------------------------------------------------- split


def test_disease_split_is_stable_and_close_to_60_20_20() -> None:
    ids = [f"ORPHA:{i}" for i in range(20_000)]
    splits = [assign_split(d) for d in ids]

    assert splits == [assign_split(d) for d in ids]
    for name, share in (("train", 0.6), ("validation", 0.2), ("test", 0.2)):
        assert splits.count(name) / len(ids) == pytest.approx(share, abs=0.015)


def test_pair_key_is_order_independent() -> None:
    assert pair_key("ORPHA:2", "ORPHA:1") == pair_key("ORPHA:1", "ORPHA:2")


# --------------------------------------------------------------------------- encoders are inductive


def test_item_idf_ignores_held_out_diseases() -> None:
    train = [{"items": {"a": 1.0, "b": 1.0}}, {"items": {"a": 1.0}}, {"items": {"c": 1.0}}]
    held_out = [{"items": {"b": 1.0}}] * 5

    def fit(catalogue):
        return WeightedItemEncoder("items", lambda r: r["items"], str).fit(train, catalogue)

    alone, with_held_out = fit(train), fit(train + held_out)
    for item in ("a", "b", "c"):
        assert alone.idf_[alone.index_[item]] == pytest.approx(with_held_out.idf_[with_held_out.index_[item]])


def test_text_vocabulary_ignores_held_out_documents() -> None:
    train = [{"d": "alpha beta"}, {"d": "alpha gamma"}, {"d": "delta"}]
    held_out = [{"d": "delta"}, {"d": "gamma"}]
    encoder = TextEncoder("text", lambda r: r["d"], ngram_range=(1, 1), min_df=2, max_df=1.0).fit(train, train + held_out)

    # "delta" and "gamma" occur once among training documents, so a held-out
    # query must not match on them, exactly as a brand-new disease would not.
    assert list(encoder.terms_) == ["alpha"]
    assert encoder.transform(held_out).nnz == 0


def test_transform_is_row_independent() -> None:
    train = [{"items": {"a": 2.0, "b": 1.0}}, {"items": {"b": 1.0}}, {"items": {"c": 1.0}}]
    encoder = WeightedItemEncoder("items", lambda r: r["items"], str).fit(train, train)
    batch = encoder.transform(train[:2] + [{"items": {"a": 1.0}}]).toarray()
    single = encoder.transform([train[0]]).toarray()

    np.testing.assert_allclose(batch[0], single[0])
    np.testing.assert_allclose(np.linalg.norm(batch, axis=1), [1.0, 1.0, 1.0], rtol=1e-6)


# --------------------------------------------------------------------------- similarity and masking


def toy_engine() -> SimilarityEngine:
    phenotype = sp.csr_matrix(np.array([[1, 0], [0.6, 0.8], [0, 1], [0, 0]], dtype=np.float32))
    gene = sp.csr_matrix(np.array([[1, 0], [1, 0], [0, 0], [0, 1]], dtype=np.float32))
    return SimilarityEngine({"phenotype": phenotype, "gene": gene}, ["d0", "d1", "d2", "d3"])


def test_block_masks_modalities_and_marks_missing_annotations() -> None:
    engine = toy_engine()
    rows, available = engine.rows(["d0"])
    gallery = engine.gallery(engine.ids)

    S, A = engine.block(rows, available, gallery)
    np.testing.assert_allclose(S[0, :, 0], [1.0, 0.6, 0.0, 0.0], atol=1e-6)
    np.testing.assert_array_equal(A[0, :, 0], [True, True, True, False])
    np.testing.assert_array_equal(A[0, :, 1], [True, True, False, True])

    S, A = engine.block(rows, available, gallery, masked=["gene"])
    assert not A[..., 1].any() and not S[..., 1].any()


def test_pair_features_agree_with_block() -> None:
    engine = toy_engine()
    S_pairs, A_pairs = engine.pair_features(["d1"] * 4, engine.ids, masked=["phenotype"])
    rows, available = engine.rows(["d1"])
    S_block, A_block = engine.block(rows, available, engine.gallery(engine.ids), masked=["phenotype"])

    np.testing.assert_allclose(S_pairs, S_block[0])
    np.testing.assert_array_equal(A_pairs, A_block[0])


# --------------------------------------------------------------------------- fusion


def collinear_training_set(seed: int = 0) -> TrainingSet:
    rng = np.random.default_rng(seed)
    n = 4000
    x1 = rng.random(n)
    x2 = np.clip(x1 + rng.normal(0, 0.05, n), 0, 1)
    x3 = rng.random(n)
    logits = -3 + 9 * x1 - 3 * x2 + 2 * x3
    y = (rng.random(n) < 1 / (1 + np.exp(-logits))).astype(np.int64)
    S = np.stack([x1, x2, x3], axis=1).astype(np.float32)
    A = np.ones_like(S, dtype=bool)
    return TrainingSet(S=S, A=A, y=y, weight=np.ones(n), benchmark=np.zeros(n, dtype=np.int64))


def test_nonnegative_fusion_has_no_negative_similarity_weights() -> None:
    data = collinear_training_set()
    modalities = ("a", "b", "c")

    free = fit_logistic(data, modalities, C=10.0, use_availability=False, nonnegative=False)
    bounded = fit_logistic(data, modalities, C=10.0, use_availability=False, nonnegative=True)

    assert free.similarity_weights.min() < 0
    assert bounded.similarity_weights.min() >= 0
    assert bounded.similarity_weights[2] > 0


def test_bounded_optimizer_reaches_the_logistic_optimum_when_unconstrained() -> None:
    data = collinear_training_set(1)
    X = data.S.astype(np.float64)
    C = 1.0

    def objective(w, b):
        z = X @ w + b
        return 0.5 * w @ w + C * np.sum(np.logaddexp(0, z) - data.y * z)

    coef, intercept = _bounded_logistic(X, data.y, data.weight, C, n_nonnegative=0)
    reference = fit_logistic(data, ("a", "b", "c"), C=C, use_availability=False, nonnegative=False)
    assert objective(coef, intercept) <= objective(reference.similarity_weights.astype(np.float64), reference.intercept) + 1e-3


def test_pattern_fusion_moves_weight_to_available_correlated_modality() -> None:
    data = collinear_training_set(2)
    modalities = ("a", "b", "c")
    base = fit_logistic(data, modalities, C=1.0, use_availability=True)
    pattern = PatternFusion(data, base)

    without_a = pattern.submodel(["a"])
    assert without_a.similarity_weights[0] == 0.0
    assert without_a.similarity_weights[1] > base.similarity_weights[1]

    # A query lacking modality "a" is scored by that submodel.
    S = np.array([[[0.0, 0.9, 0.2], [0.0, 0.1, 0.8]]], dtype=np.float32)
    A = np.array([[[False, True, True], [False, True, True]]])
    np.testing.assert_allclose(pattern.score(S, A), without_a.score(S, A), rtol=1e-6)


def test_linear_fusion_contributions_add_up_to_the_logit() -> None:
    fusion = LinearFusion(("a", "b"), np.array([2.0, 1.0], np.float32), np.array([0.5, -0.25], np.float32), -1.0)
    S = np.array([0.3, 0.6], np.float32)
    A = np.array([True, True])
    total = fusion.intercept + fusion.similarity_contributions(S).sum() + fusion.availability_adjustment(A)
    assert float(fusion.score(S[None, None], A[None, None])[0, 0]) == pytest.approx(float(total))


# --------------------------------------------------------------------------- integration


@pytest.fixture(scope="module")
def bundle():
    if not HAS_FEATURES:
        pytest.skip("feature tables not built (run generate_features.py)")
    from data_sources import load_bundle

    return load_bundle()


def test_benchmark_masks_cover_each_label_source(bundle) -> None:
    from benchmarks import BENCHMARK_SPECS

    assert {"ontology", "name"} <= set(BENCHMARK_SPECS["orphanet_siblings"].masked)
    assert {"gene", "pathway", "ot_gene"} <= set(BENCHMARK_SPECS["shared_causal_gene"].masked)
    assert {"drug", "drug_target", "ot_gene"} <= set(BENCHMARK_SPECS["shared_drug"].masked)


def test_shared_drug_pairs_are_distinct_indications(bundle) -> None:
    from benchmarks import build_relations

    relation = build_relations(bundle)["shared_drug"]
    trial = bundle.labels.trial_drugs
    for (a, b), reason in list(relation.evidence.items())[:5000]:
        drug = reason.rsplit("(", 1)[-1].rstrip(")")
        assert trial[a][drug] != trial[b][drug]


def test_user_input_is_validated(bundle) -> None:
    from data_sources import record_from_user_input

    record, warnings = record_from_user_input(
        {"name": "x", "phenotypes": ["HP:0001250", "HP:9999999"], "genes": ["CDKL5"], "drugs": ["not-a-drug-xyz"]},
        bundle.knowledge,
    )
    assert record["phenotypes"] == {"HP:0001250": 0.5}
    assert record["genes"] == {"CDKL5": 1.0}
    assert len(warnings) == 2

    record, warnings = record_from_user_input(
        {"name": "x", "inheritance": ["AR", "X-linked dominant", "sporadic"], "onset": ["infancy", "toddler"],
         "prevalence": "1-9 / 100 000"},
        bundle.knowledge,
    )
    assert record["inheritance"] == ["autosomal recessive", "x-linked dominant"]
    assert record["onset"] == ["Infancy"]
    assert record["prevalence"] is None
    assert len(warnings) == 3


def test_new_disease_is_placed_next_to_its_gene_relatives() -> None:
    from production import MODEL_PATH, load_model

    if not MODEL_PATH.is_file():
        pytest.skip("production model not built (run build_graph.py)")
    from data_sources import record_from_user_input
    from place_disease import place

    model = load_model()
    document = json.loads((V2_DIR / "examples" / "new_disease_example.json").read_text(encoding="utf-8"))
    record, _ = record_from_user_input(document, model.knowledge)
    results, scored = place(model, record, document["id"], top=10)

    assert len(results) == 10
    assert all(r["explanation"] for r in results)
    assert scored["fusion"].similarity_weights[list(model.engine.modalities).index("gene")] > 0
    cdkl5 = {"ORPHA:505652", "ORPHA:1934", "ORPHA:3095", "ORPHA:697160"}
    assert cdkl5 & {r["id"] for r in results}
