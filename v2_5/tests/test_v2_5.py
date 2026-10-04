from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "v2"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from modalities import _pathway_items
from v2_5.benchmark_medgemma import extraction_path, load_extraction
from v2_5.client import (
    ClientConfig,
    Completion,
    EndpointUnavailable,
    InvalidModelResponse,
    MedGemmaClient,
    extract_json_document,
)
from v2_5.extraction import EXTRACTION_SCHEMA, HybridPaperExtractor, save_extraction
from v2_5.passages import SourceUnit, select_passages, units_from_text
from v2_5.transformers_client import TransformersConfig, TransformersMedGemmaClient


class FakeKnowledge:
    hpo_labels = {"HP:1": "Muscle weakness", "HP:2": "Short stature"}
    hpo_alternatives = {}
    gene_symbols = frozenset({"ABCD1", "TP53"})
    gene_pathways = {"ABCD1": frozenset({"R-HSA-1"})}
    pathway_labels = {"R-HSA-1": "Fatty acid beta oxidation"}
    drug_labels = {"CHEMBL1": "Riluzole"}
    drug_targets = {"CHEMBL1": frozenset({"SOD1"})}
    drug_parent = {}

    @staticmethod
    def canonical_hpo(value: str) -> str | None:
        return value if value in FakeKnowledge.hpo_labels else None

    @staticmethod
    def is_phenotypic_abnormality(value: str) -> bool:
        return value in FakeKnowledge.hpo_labels

    @staticmethod
    def resolve_drug(value: str) -> str | None:
        return "CHEMBL1" if value.casefold() in {"chembl1", "riluzole"} else None


class StubClient:
    def __init__(self, document: dict):
        self.document = document

    def complete_json(self, *args, **kwargs) -> Completion:
        return Completion(
            document=self.document,
            model="medgemma-test",
            model_revision="test-revision",
            response_sha256="response",
            cached=False,
            usage={"prompt_tokens": 100},
            cache_key="cache",
        )


class FailingClient:
    def complete_json(self, *args, **kwargs):
        raise EndpointUnavailable("offline")


def empty_document() -> dict:
    return {
        "focal_disease_label": "Example disease",
        "phenotypes": [],
        "genes": [],
        "pathways": [],
        "drugs": [],
        "inheritance": [],
        "onset": [],
        "prevalence": [],
    }


def evidence(**values):
    return {
        "locator": "[P0001 | Paper]",
        "quote": values.pop("quote"),
        "confidence": values.pop("confidence", 3),
        "assertion": values.pop("assertion", "present"),
        **values,
    }


def test_hybrid_extraction_requires_grounded_verbatim_evidence() -> None:
    document = empty_document()
    document["phenotypes"] = [
        evidence(term="proximal weakness", hpo_id="HP:1", quote="muscle weakness"),
        evidence(term="short stature", hpo_id="HP:2", quote="not in the paper"),
    ]
    document["genes"] = [
        evidence(symbol="ABCD1", relationship="causal", quote="ABCD1"),
        evidence(symbol="TP53", relationship="associated", quote="TP53", assertion="comparator"),
    ]
    document["pathways"] = [
        evidence(term="Fatty acid beta oxidation", pathway_id="R-HSA-1", quote="beta oxidation")
    ]
    document["drugs"] = [
        evidence(name="Riluzole", chembl_id="CHEMBL1", role="treatment", quote="Riluzole")
    ]
    document["inheritance"] = [
        evidence(value="autosomal recessive", quote="autosomal recessive")
    ]
    document["onset"] = [evidence(value="Infancy", quote="infancy")]
    document["prevalence"] = [
        evidence(value_per_person=1e-5, population="", quote="1 in 100,000")
    ]
    text = (
        "ABCD1 causes muscle weakness through beta oxidation. Riluzole was used. "
        "The condition is autosomal recessive, begins in infancy, and affects 1 in 100,000. "
        "TP53 was present only in a comparator disease."
    )
    result = HybridPaperExtractor(FakeKnowledge(), StubClient(document)).extract(text)
    assert result.status == "hybrid_success"
    assert result.record["phenotypes"] == {"HP:1": 1.0}
    assert result.record["genes"]["ABCD1"] == 1.0
    assert "TP53" not in result.record["genes"]
    assert result.record["pathways"] == {"R-HSA-1": 1.0}
    assert result.record["drugs"]["CHEMBL1"] == 1.0
    assert result.record["inheritance"] == ["autosomal recessive"]
    assert result.record["onset"] == ["Infancy"]
    assert result.record["prevalence"] == 1e-5
    assert len(result.accepted_evidence) == 7
    assert len(result.rejected_features) == 2
    assert all(item["verification_status"] == "unverified" for item in result.accepted_evidence)


def test_hybrid_extraction_tolerates_null_and_non_array_fields() -> None:
    document = empty_document()
    document["prevalence"] = None
    document["onset"] = {"value": "Infancy"}
    result = HybridPaperExtractor(FakeKnowledge(), StubClient(document)).extract("A disease paper")
    assert result.status == "hybrid_success"
    assert result.record["prevalence"] is None
    assert result.record["onset"] == []
    assert result.rejected_features[0]["reason"] == "field is not an array"


def test_failed_model_falls_back_unless_strict() -> None:
    result = HybridPaperExtractor(FakeKnowledge(), FailingClient()).extract("ABCD1 causes muscle weakness.")
    assert result.status == "deterministic_fallback"
    assert result.record["genes"] == {"ABCD1": 1.0}
    assert result.warnings == ["offline"]
    with pytest.raises(EndpointUnavailable):
        HybridPaperExtractor(FakeKnowledge(), FailingClient(), strict_llm=True).extract("ABCD1")


def test_passage_selection_respects_budget_and_keeps_high_value_text() -> None:
    units = [
        SourceUnit("References", "r", "citation " * 100),
        SourceUnit("Title", "t", "A rare genetic disorder"),
        SourceUnit("Results", "p", "ABCD1 pathogenic variants cause infantile muscle weakness. " * 20),
    ]
    passages = select_passages(units, max_words=50)
    assert sum(item.word_count for item in passages) <= 50
    assert any("ABCD1" in item.text for item in passages)
    assert all("citation" not in item.text for item in passages)


def test_client_rejects_remote_endpoint_by_default() -> None:
    with pytest.raises(ValueError, match="non-local"):
        MedGemmaClient(ClientConfig(base_url="https://example.org/v1"))
    MedGemmaClient(ClientConfig(base_url="http://localhost:8000/v1"))


def test_json_response_parser_handles_fences_and_reasoning() -> None:
    assert extract_json_document('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json_document('analysis first\n{"a": 2}\nend') == {"a": 2}


def test_client_cache_avoids_duplicate_endpoint_calls(tmp_path: Path) -> None:
    class CountingClient(MedGemmaClient):
        calls = 0

        def _request(self, url, payload=None):
            self.calls += 1
            return {
                "model": "test",
                "choices": [{"message": {"content": json.dumps(empty_document())}}],
                "usage": {"prompt_tokens": 1},
            }

    client = CountingClient(ClientConfig(cache_dir=tmp_path))
    first = client.complete_json("system", "user", EXTRACTION_SCHEMA, "v1")
    second = client.complete_json("system", "user", EXTRACTION_SCHEMA, "v1")
    assert client.calls == 1
    assert not first.cached and second.cached
    assert first.document == second.document


def test_explicit_pathway_is_merged_with_gene_derived_pathways() -> None:
    record = {
        "genes": {"ABCD1": 0.75},
        "pathways": {"R-HSA-2": 1.0},
    }
    knowledge = FakeKnowledge()
    knowledge.pathway_labels = {"R-HSA-1": "One", "R-HSA-2": "Two"}
    assert _pathway_items(knowledge, record) == {"R-HSA-1": 0.75, "R-HSA-2": 1.0}


def test_extraction_files_are_resumable(tmp_path: Path) -> None:
    result = HybridPaperExtractor(FakeKnowledge(), StubClient(empty_document())).extract("A disease paper")
    path = extraction_path(tmp_path, "ORPHA:1", "title")
    save_extraction(path, result, provenance={"paper_id": 1})
    loaded = load_extraction(path)
    assert loaded.source_sha256 == result.source_sha256
    assert loaded.record == result.record


def test_plain_text_units_are_stable() -> None:
    units = units_from_text("A title\n\nAn abstract.\n\nA result.")
    assert [unit.section for unit in units] == ["Title", "Abstract", "Paper"]


def test_transformers_backend_is_lazy() -> None:
    client = TransformersMedGemmaClient(TransformersConfig())
    assert not client.loaded
    assert client._torch is None
    assert client.config.quantization == "4bit"


def test_transformers_backend_generation_contract_without_gpu(tmp_path: Path) -> None:
    class FakeInputs(dict):
        def to(self, _device):
            return self

    class FakeProcessor:
        messages = None
        template_kwargs = None

        @classmethod
        def apply_chat_template(cls, messages, **kwargs):
            cls.messages = messages
            cls.template_kwargs = kwargs
            return FakeInputs(input_ids=np.zeros((1, 3), dtype=np.int64))

        @staticmethod
        def decode(*args, **kwargs):
            return json.dumps(empty_document())[1:]

    class FakeModel:
        device = "cuda:0"

        @staticmethod
        def generate(**kwargs):
            return np.zeros((1, 8), dtype=np.int64)

    class FakeCuda:
        class OutOfMemoryError(RuntimeError):
            pass

        @staticmethod
        def empty_cache():
            pass

    class FakeTorch:
        cuda = FakeCuda()

        @staticmethod
        def inference_mode():
            return nullcontext()

    client = TransformersMedGemmaClient(
        TransformersConfig(response_cache_dir=tmp_path, max_output_tokens=5)
    )
    client._model = FakeModel()
    client._processor = FakeProcessor()
    client._torch = FakeTorch()
    result = client.complete_json("system", "user", EXTRACTION_SCHEMA, "v1", use_cache=False)
    assert result.document == empty_document()
    assert result.usage["prompt_tokens"] == 3
    assert result.usage["completion_tokens"] == 5
    assert FakeProcessor.messages[-1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "{"}],
    }
    assert FakeProcessor.template_kwargs["add_generation_prompt"] is False
    assert FakeProcessor.template_kwargs["continue_final_message"] is True

    FakeProcessor.decode = staticmethod(lambda *args, **kwargs: '"outer": {"inner": 1}')
    with pytest.raises(InvalidModelResponse, match="complete top-level JSON"):
        client.complete_json("system", "user", EXTRACTION_SCHEMA, "v2", use_cache=False)

