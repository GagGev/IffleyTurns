from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

from benchmark_paper_input import (
    PaperExtractor,
    QueryPaper,
    TextUnit,
    mask_spans,
    paper_text,
    ranking_metrics,
    truncate_words,
)


class FakeKnowledge:
    gene_symbols = {"ABCD1", "TP53"}
    hpo_labels = {"HP:1": "Muscle weakness", "HP:2": "Short stature"}
    drug_labels = {"CHEMBL1": "Riluzole", "CHEMBL2": "Aspirin"}

    @staticmethod
    def is_phenotypic_abnormality(identifier: str) -> bool:
        return identifier.startswith("HP:")


def query_paper() -> QueryPaper:
    return QueryPaper(
        target_id="ORPHA:1",
        paper_id=1,
        pmid="1",
        pmcid="PMC1",
        doi="",
        title="Alpha syndrome and beta disease",
        abstract="Alpha syndrome causes weakness.",
        year=2025,
        source_urls=[],
        full_text_path="x.xml",
        title_mention=True,
        metadata_target_count=2,
        paper_comparators={"ORPHA:2"},
        body_chars=20,
        units=[TextUnit("Results", "body/p[1]", "ABCD1 causes muscle weakness.")],
        mentions={
            "title": [(0, 14), (19, 31)],
            "abstract": [(0, 14)],
        },
    )


def test_mask_spans_merges_overlap_and_masks_literal_orpha() -> None:
    assert mask_spans("Alpha syndrome ORPHA:123", [(0, 5), (0, 14)]) == " DISEASE   DISEASE "


def test_information_levels_are_nested_and_masked() -> None:
    paper = query_paper()
    assert "Alpha syndrome" not in paper_text(paper, "title_abstract", masked=True)
    assert "beta disease" not in paper_text(paper, "title", masked=True)
    assert "ABCD1" in paper_text(paper, "full_text", masked=True)
    assert paper_text(paper, "title", masked=False) == paper.title
    assert len(truncate_words("one two three four", 3).split()) == 3


def test_structured_extractor_uses_paper_text_only() -> None:
    extractor = PaperExtractor(FakeKnowledge())
    record, counts = extractor.extract(
        "Autosomal recessive ABCD1 disease has muscle weakness. "
        "Riluzole was used. Prevalence is 1 in 100,000 with infantile onset."
    )
    assert record["genes"] == {"ABCD1": 1.0}
    assert record["phenotypes"] == {"HP:1": 0.5}
    assert record["drugs"] == {"CHEMBL1": 1.0}
    assert record["inheritance"] == ["autosomal recessive"]
    assert record["onset"] == ["Infancy"]
    assert math.isclose(record["prevalence"], 1e-5)
    assert all(counts[field] == 1 for field in counts)
    assert record["name"] == ""
    assert record["ontology_parents"] == []


def test_text_only_extraction_has_no_structured_identity() -> None:
    record, counts = PaperExtractor(FakeKnowledge()).extract("ABCD1 muscle weakness", structured=False)
    assert record["description"] == "ABCD1 muscle weakness"
    assert not record["genes"] and not record["phenotypes"]
    assert sum(counts.values()) == 0


def test_ranking_metrics() -> None:
    metrics = ranking_metrics(np.array([0.9, 0.8, 0.1]), {0, 2})
    assert math.isclose(metrics["map"], (1.0 + 2.0 / 3.0) / 2.0)
    assert metrics["mrr"] == 1.0
    assert metrics["hits_at_10"] == 1.0
    assert metrics["recall_at_10"] == 1.0
