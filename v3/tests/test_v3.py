"""Tests for v3: regulatory ground truth, leakage guards, features and models.

Unit tests use small synthetic inputs.  Integration tests use the cached
world (``v3/.data/cache/world.pkl``) and are skipped when it has not been built.

Run:  python -m pytest v3/tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

V3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3_DIR))

from common import CACHE_DIR  # noqa: E402
from regulatory import (  # noqa: E402
    DiseaseMatcher,
    OrphanetCatalog,
    build_relations,
    loose_key,
    normalize_text,
)

HAS_WORLD = (CACHE_DIR / "world.pkl").is_file()


# --------------------------------------------------------------------------- disease matching


def small_catalog() -> OrphanetCatalog:
    names = {
        "ORPHA:1": "Proximal spinal muscular atrophy",
        "ORPHA:2": "Small cell lung cancer",
        "ORPHA:3": "Fabry disease",
        "ORPHA:4": "Huntington disease",
        "ORPHA:5": "Cystic fibrosis",
        "ORPHA:6": "Beta-thalassemia intermedia",
        "ORPHA:10": "Lysosomal disease",
        "ORPHA:11": "Gaucher disease",
    }
    return OrphanetCatalog(
        names=names,
        diseases=frozenset(set(names) - {"ORPHA:10"}),
        groups=frozenset({"ORPHA:10"}),
        excluded=frozenset(),
        children={"ORPHA:10": {"ORPHA:3", "ORPHA:11"}},
    )


def small_matcher() -> DiseaseMatcher:
    catalog = small_catalog()
    preferred = {o: {name} for o, name in catalog.names.items()}
    unmodified = {"ORPHA:1": {"spinal muscular atrophy"}}
    return DiseaseMatcher(catalog, [preferred, {}, {}, unmodified], heads=frozenset())


def test_normalize_text_handles_spelling_possessives_and_roman_numerals() -> None:
    assert normalize_text("Treatment of Huntington's Disease") == normalize_text("treatment of huntington disease")
    assert "4" in normalize_text("glycogen storage disease type IV").split()
    assert normalize_text("oesophageal tumour") == normalize_text("esophageal tumor")


def test_loose_key_ignores_order_fillers_and_plurals() -> None:
    assert loose_key(normalize_text("sarcomas of soft tissue")) == loose_key(normalize_text("soft tissue sarcoma"))
    assert loose_key("huntingtons") == loose_key("huntington")


def test_matcher_maps_exact_modifier_and_qualified_indications() -> None:
    matcher = small_matcher()
    assert matcher.map_text("Treatment of Fabry disease")[0] == {"ORPHA:3"}
    assert matcher.map_text("Treatment of spinal muscular atrophy")[0] == {"ORPHA:1"}
    ids, tier, _ = matcher.map_text("Treatment of pulmonary infection in patients with cystic fibrosis")
    assert ids == {"ORPHA:5"} and tier in {"qualifier", "contained"}


def test_contained_match_respects_negation() -> None:
    matcher = small_matcher()
    ids, _, _ = matcher.map_text("Treatment of non-small cell lung cancer")
    assert "ORPHA:2" not in ids


def test_list_tier_requires_every_multiword_piece() -> None:
    matcher = small_matcher()
    ids, tier, _ = matcher.map_text("Treatment of Fabry disease and Huntington disease")
    assert (ids, tier) == ({"ORPHA:3", "ORPHA:4"}, "list")
    _, tier, _ = matcher.map_text("Treatment of acute intermittent porphyria and Fabry disease")
    assert tier != "list"


# --------------------------------------------------------------------------- relations


def designations(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"record_id": r, "drug_id": d, "orpha_ids": o, "date": pd.Timestamp(t), "source": "FDA", "approved": a}
            for r, d, o, t, a in rows
        ]
    )


def test_relations_need_distinct_records_and_take_the_later_date() -> None:
    frame = designations(
        [
            ("FDA:1", "DRUG:A", ["ORPHA:3"], "2001-05-01", True),
            ("FDA:2", "DRUG:A", ["ORPHA:5"], "2009-03-01", False),
            ("FDA:3", "DRUG:B", ["ORPHA:4", "ORPHA:5"], "2005-01-01", False),
        ]
    )
    relations, broad = build_relations(frame, small_catalog())
    pairs = {(r.a, r.b): r for r in relations.itertuples(index=False)}
    assert set(pairs) == {("ORPHA:3", "ORPHA:5")}
    assert pairs[("ORPHA:3", "ORPHA:5")].date == pd.Timestamp("2009-03-01")
    assert not pairs[("ORPHA:3", "ORPHA:5")].approved_both
    assert not broad


def test_relations_skip_nested_pairs_and_broad_drugs() -> None:
    rows = [
        ("FDA:1", "DRUG:A", ["ORPHA:10"], "2001-01-01", False),
        ("FDA:2", "DRUG:A", ["ORPHA:3"], "2002-01-01", False),
    ]
    rows += [(f"EMA:{i}", "DRUG:X", [f"ORPHA:{100 + i}"], "2003-01-01", False) for i in range(3)]
    relations, broad = build_relations(designations(rows), small_catalog(), max_diseases_per_drug=2)
    assert relations.empty
    assert broad == frozenset({"DRUG:X"})


# --------------------------------------------------------------------------- inputs


def test_drug_vocabulary_keeps_drug_names_and_drops_biomedical_words() -> None:
    from modalities import drug_vocabulary

    knowledge = SimpleNamespace(
        hpo_labels={"HP:1": "Abnormal circulating glutamine concentration"},
        ontology_labels={"X:1": "Factor VII deficiency"},
        drug_name_index={"imatinib": "CHEMBL941"},
    )
    words = drug_vocabulary(["Imatinib mesylate", "glutamine", "recombinant factor VIIa", "cells resulting from culture"], knowledge)
    assert "imatinib" in words
    assert not {"glutamine", "factor", "cells", "resulting", "culture"} & words


def test_pair_head_is_symmetric() -> None:
    import torch

    from neural import DEFAULT_CONFIG, StaticNet

    torch.manual_seed(0)
    net = StaticNet({"phenotype": 5, "text": 7}, DEFAULT_CONFIG).eval()
    za, zb = torch.randn(4, DEFAULT_CONFIG["z_dim"]), torch.randn(4, DEFAULT_CONFIG["z_dim"])
    S, A = torch.rand(4, 2), torch.ones(4, 2)
    assert torch.allclose(net.pair(za, zb, S, A), net.pair(zb, za, S, A), atol=1e-5)


def test_nonnegative_logistic_keeps_similarity_weights_nonnegative() -> None:
    from baselines import fit_nonnegative_logistic

    rng = np.random.default_rng(0)
    S = rng.random((400, 3)).astype(np.float32)
    y = (S[:, 0] + 0.1 * rng.standard_normal(400) > 0.6).astype(float)
    S[:, 1] = 1.0 - S[:, 0]
    fusion = fit_nonnegative_logistic(S, np.ones_like(S, dtype=bool), y, C=1.0)
    assert (fusion.similarity_weights >= 0).all()
    assert fusion.similarity_weights[0] > fusion.similarity_weights[1]


# --------------------------------------------------------------------------- integration: leakage guards


@pytest.fixture(scope="module")
def world():
    if not HAS_WORLD:
        pytest.skip("world cache not built (run python v3/run_evaluation.py)")
    from world import load_world

    return load_world()


def test_snapshot_contains_only_knowledge_before_its_date(world) -> None:
    cutoff = pd.Timestamp("2014-01-01")
    snapshot = world.snapshot(cutoff)
    assert (snapshot.relations["date"] < cutoff).all()
    history = world.regulatory.history
    for disease, drugs in snapshot.drugs.items():
        assert all(history[disease][drug] < cutoff for drug in drugs)
    assert np.nanmax(snapshot.first_designation) < 2014.0


def test_task_labels_start_at_the_cutoff_and_exclude_known_and_nested(world) -> None:
    from world import make_task

    cutoff, until = pd.Timestamp("2014-01-01"), pd.Timestamp("2018-01-01")
    task = make_task(world, "t", cutoff, until)
    assert ((task.relations["date"] >= cutoff) & (task.relations["date"] < until)).all()
    queries = task.queries(np.arange(world.n), world)[:50]
    excluded = task.excluded(queries, world).toarray()
    known = task.snapshot.adjacency[queries].toarray() > 0
    assert (excluded[known]).all()
    assert excluded[np.arange(len(queries)), queries].all()
    assert not ((task.positives[queries].toarray() > 0) & known).any()


def test_designated_groups_are_not_the_only_groups(world) -> None:
    designated = {o for ids in world.regulatory.designations["orpha_ids"] for o in ids}
    groups = {world.ids[i] for i in np.flatnonzero(world.is_group)}
    assert len(groups - designated) > len(groups & designated)


def test_new_disease_rows_reproduce_node_features(world) -> None:
    from features import StaticSimilarity, TemporalFeatures

    snapshot = world.snapshot(pd.Timestamp("2018-01-01"))
    sim = StaticSimilarity.from_world(world)
    features = TemporalFeatures(world, snapshot, sim, None)
    node = int(np.flatnonzero(snapshot.warm)[0])
    record = {**world.bundle.records[world.ids[node]], "drugs": snapshot.drugs[world.ids[node]]}
    as_node = features.block(np.array([node]), groups=("mechanism",))
    as_new = features.block_for(features.new_rows([record]), None, None, None, groups=("mechanism",))
    for name in as_node:
        np.testing.assert_allclose(as_node[name], as_new[name], atol=1e-6)
    S_node, A_node = sim.block(np.array([node]))
    encoded = world.encoder.transform([world.bundle.records[world.ids[node]]])
    available = np.array([[encoded[m].nnz > 0 for m in sim.modalities]])
    S_new, A_new = sim.block(query_matrices=encoded, query_available=available)
    np.testing.assert_allclose(S_node, S_new, atol=1e-5)
    assert (A_node == A_new).all()


def test_description_text_has_no_designated_drug_words(world) -> None:
    from modalities import TOKEN

    encoder = world.encoder.encoders["text"]
    vocabulary = {t for term in encoder.terms_ for t in TOKEN.findall(str(term))}
    assert not vocabulary & world.drug_words
