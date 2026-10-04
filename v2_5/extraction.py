"""Hybrid deterministic + MedGemma extraction into the v2 disease schema."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "v2"
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

from benchmark_paper_input import PaperExtractor
from data_sources import Knowledge, empty_record
from modalities import INHERITANCE_MAP, ONSET_BINS

from .client import Completion, JsonCompletionClient, MedGemmaError
from .passages import Passage, SourceUnit, format_passages, select_passages, units_from_text


PROMPT_VERSION = "medgemma-paper-extraction-2.5.0"
WEIGHTS = {1: 0.5, 2: 0.75, 3: 1.0}
ASSERTIONS = {"present", "absent", "uncertain", "comparator"}
GENE_RELATIONS = {"causal", "associated", "therapeutic_target", "unclear"}
DRUG_ROLES = {"treatment", "investigational", "supportive", "response", "mentioned", "comparator"}
SPACE = re.compile(r"\s+")


def _evidence_properties() -> dict[str, Any]:
    return {
        "locator": {"type": "string"},
        "quote": {"type": "string"},
        "confidence": {"type": "integer", "minimum": 1, "maximum": 3},
    }


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _array(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "items": _object(properties)}


EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "focal_disease_label": {"type": "string"},
        "phenotypes": _array(
            {
                "term": {"type": "string"},
                "hpo_id": {"type": ["string", "null"]},
                "assertion": {"type": "string", "enum": sorted(ASSERTIONS)},
                **_evidence_properties(),
            }
        ),
        "genes": _array(
            {
                "symbol": {"type": "string"},
                "relationship": {"type": "string", "enum": sorted(GENE_RELATIONS)},
                "assertion": {"type": "string", "enum": sorted(ASSERTIONS)},
                **_evidence_properties(),
            }
        ),
        "pathways": _array(
            {
                "term": {"type": "string"},
                "pathway_id": {"type": ["string", "null"]},
                "assertion": {"type": "string", "enum": sorted(ASSERTIONS)},
                **_evidence_properties(),
            }
        ),
        "drugs": _array(
            {
                "name": {"type": "string"},
                "chembl_id": {"type": ["string", "null"]},
                "role": {"type": "string", "enum": sorted(DRUG_ROLES)},
                "assertion": {"type": "string", "enum": sorted(ASSERTIONS)},
                **_evidence_properties(),
            }
        ),
        "inheritance": _array(
            {
                "value": {"type": "string", "enum": sorted(INHERITANCE_MAP)},
                "assertion": {"type": "string", "enum": sorted(ASSERTIONS)},
                **_evidence_properties(),
            }
        ),
        "onset": _array(
            {
                "value": {"type": "string", "enum": list(ONSET_BINS) + ["All ages"]},
                "assertion": {"type": "string", "enum": sorted(ASSERTIONS)},
                **_evidence_properties(),
            }
        ),
        "prevalence": _array(
            {
                "value_per_person": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "maximum": 1,
                },
                "population": {"type": "string"},
                "assertion": {"type": "string", "enum": sorted(ASSERTIONS)},
                **_evidence_properties(),
            }
        ),
    },
    "required": [
        "focal_disease_label",
        "phenotypes",
        "genes",
        "pathways",
        "drugs",
        "inheritance",
        "onset",
        "prevalence",
    ],
    "additionalProperties": False,
}


SYSTEM_PROMPT = """You extract structured facts about the focal rare disease from scientific-paper passages.

Rules:
- Extract only facts explicitly asserted for the focal disease or its patient cohort.
- Exclude facts about comparator diseases, family members without the disease, background examples, methods, and cited prior diseases.
- Do not treat negated, ruled-out, speculative, or uncertain findings as present.
- Do not infer a gene, phenotype, drug, pathway, onset, inheritance, or prevalence that is not stated.
- Every item must include one supplied passage locator and a short verbatim quote copied from that passage.
- Use confidence 3 for direct unambiguous assertions, 2 for clear but less specific assertions, and 1 for weakly contextual assertions.
- Return JSON only and obey the schema. Empty arrays are preferred to guesses.
- Output is automated and will be independently validated."""


@dataclass
class AcceptedEvidence:
    feature_type: str
    identifier: str
    label: str
    locator: str
    quote: str
    confidence: int
    extraction_method: str = "medgemma"
    verification_status: str = "unverified"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RejectedFeature:
    feature_type: str
    value: str
    reason: str
    item: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    record: dict[str, Any]
    focal_disease_label: str
    passages: list[dict[str, Any]]
    accepted_evidence: list[dict[str, Any]]
    rejected_features: list[dict[str, Any]]
    warnings: list[str]
    status: str
    model: str
    model_revision: str
    prompt_version: str
    response_sha256: str
    cache_key: str
    cached: bool
    usage: dict[str, Any]
    source_id: str
    source_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, document: dict[str, Any]) -> "ExtractionResult":
        return cls(**document)


def _normalized(value: str) -> str:
    return SPACE.sub(" ", str(value)).strip().casefold()


def _unique_label_index(labels: dict[str, str]) -> dict[str, str]:
    candidates: dict[str, set[str]] = {}
    for identifier, label in labels.items():
        candidates.setdefault(_normalized(label), set()).add(identifier)
    return {
        label: next(iter(identifiers))
        for label, identifiers in candidates.items()
        if len(identifiers) == 1
    }


class HybridPaperExtractor:
    def __init__(
        self,
        knowledge: Knowledge,
        client: JsonCompletionClient | None,
        max_input_words: int = 2_000,
        strict_llm: bool = False,
    ):
        self.knowledge = knowledge
        self.client = client
        self.max_input_words = max_input_words
        self.strict_llm = strict_llm
        self.deterministic = PaperExtractor(knowledge)
        self.hpo_by_label = _unique_label_index(knowledge.hpo_labels)
        self.pathway_by_label = _unique_label_index(knowledge.pathway_labels)

    def _prompt(
        self,
        passages: Sequence[Passage],
        focal_hint: str,
        candidates: dict[str, Any],
    ) -> str:
        hint = focal_hint.strip() or (
            "The focal disease is the primary rare disorder studied in the title "
            "and abstract. Disease names may have been replaced with DISEASE."
        )
        candidate_summary = {
            "phenotypes": [
                {"id": identifier, "label": self.knowledge.hpo_labels.get(identifier, identifier)}
                for identifier in candidates.get("phenotypes", {})
            ],
            "genes": sorted(candidates.get("genes", {})),
            "drugs": [
                {"id": identifier, "label": self.knowledge.drug_labels.get(identifier, identifier)}
                for identifier in candidates.get("drugs", {})
            ],
            "inheritance": list(candidates.get("inheritance", [])),
            "onset": list(candidates.get("onset", [])),
            "prevalence": candidates.get("prevalence"),
        }
        return f"""Focal-disease instruction:
{hint}

Controlled inheritance values:
{", ".join(sorted(INHERITANCE_MAP))}

Controlled onset values:
{", ".join((*ONSET_BINS, "All ages"))}

Deterministically detected candidates are listed below only to help with
canonical IDs. They are not established facts. Return a candidate only when a
supplied passage explicitly asserts it for the focal disease:
{json.dumps(candidate_summary, ensure_ascii=False)}

Extract from these passages only:

{format_passages(passages)}
"""

    @staticmethod
    def _passage_map(passages: Sequence[Passage]) -> dict[str, Passage]:
        return {passage.locator: passage for passage in passages}

    def _evidence(
        self,
        item: dict[str, Any],
        passages: dict[str, Passage],
        feature_type: str,
    ) -> tuple[str, str, int] | RejectedFeature:
        locator = str(item.get("locator", "")).strip()
        quote = str(item.get("quote", "")).strip()
        try:
            confidence = int(item.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0
        if locator not in passages:
            return RejectedFeature(feature_type, str(item), "unknown evidence locator", item)
        if not quote or _normalized(quote) not in _normalized(passages[locator].text):
            return RejectedFeature(feature_type, str(item), "quote is not verbatim evidence", item)
        if confidence not in WEIGHTS:
            return RejectedFeature(feature_type, str(item), "confidence must be 1, 2, or 3", item)
        if item.get("assertion") != "present":
            return RejectedFeature(
                feature_type,
                str(item),
                f"assertion is {item.get('assertion')!r}, not present",
                item,
            )
        return locator, quote, confidence

    def _accept(
        self,
        accepted: list[AcceptedEvidence],
        feature_type: str,
        identifier: str,
        label: str,
        evidence: tuple[str, str, int],
        **metadata: Any,
    ) -> float:
        locator, quote, confidence = evidence
        accepted.append(
            AcceptedEvidence(
                feature_type=feature_type,
                identifier=identifier,
                label=label,
                locator=locator,
                quote=quote,
                confidence=confidence,
                metadata=metadata,
            )
        )
        return WEIGHTS[confidence]

    def _ground(
        self,
        document: dict[str, Any],
        passages: Sequence[Passage],
        baseline_record: dict[str, Any],
        preserve_baseline_features: bool,
    ) -> tuple[dict[str, Any], list[AcceptedEvidence], list[RejectedFeature]]:
        if preserve_baseline_features:
            record = copy.deepcopy(baseline_record)
        else:
            record = empty_record("")
            record["description"] = baseline_record["description"]
        record.setdefault("pathways", {})
        accepted: list[AcceptedEvidence] = []
        rejected: list[RejectedFeature] = []
        passage_map = self._passage_map(passages)

        for item in document.get("phenotypes", [])[:100]:
            if not isinstance(item, dict):
                rejected.append(RejectedFeature("phenotype", str(item), "item is not an object"))
                continue
            evidence = self._evidence(item, passage_map, "phenotype")
            if isinstance(evidence, RejectedFeature):
                rejected.append(evidence)
                continue
            hpo_id = self.knowledge.canonical_hpo(str(item.get("hpo_id") or ""))
            if hpo_id is None:
                hpo_id = self.hpo_by_label.get(_normalized(str(item.get("term", ""))))
            if hpo_id is None or not self.knowledge.is_phenotypic_abnormality(hpo_id):
                rejected.append(RejectedFeature("phenotype", str(item.get("term", "")), "unmapped HPO term", item))
                continue
            weight = self._accept(
                accepted,
                "phenotype",
                hpo_id,
                self.knowledge.hpo_labels.get(hpo_id, str(item.get("term", ""))),
                evidence,
            )
            record["phenotypes"][hpo_id] = max(weight, record["phenotypes"].get(hpo_id, 0.0))

        for item in document.get("genes", [])[:100]:
            if not isinstance(item, dict):
                rejected.append(RejectedFeature("gene", str(item), "item is not an object"))
                continue
            evidence = self._evidence(item, passage_map, "gene")
            if isinstance(evidence, RejectedFeature):
                rejected.append(evidence)
                continue
            symbol = str(item.get("symbol", "")).strip()
            relationship = str(item.get("relationship", "unclear"))
            if symbol not in self.knowledge.gene_symbols:
                rejected.append(RejectedFeature("gene", symbol, "unknown gene symbol", item))
                continue
            if relationship == "unclear":
                rejected.append(RejectedFeature("gene", symbol, "unclear relationship", item))
                continue
            weight = self._accept(accepted, "gene", symbol, symbol, evidence, relationship=relationship)
            field = "ot_genes" if relationship == "therapeutic_target" else "genes"
            record[field][symbol] = max(weight, record[field].get(symbol, 0.0))

        for item in document.get("pathways", [])[:100]:
            if not isinstance(item, dict):
                rejected.append(RejectedFeature("pathway", str(item), "item is not an object"))
                continue
            evidence = self._evidence(item, passage_map, "pathway")
            if isinstance(evidence, RejectedFeature):
                rejected.append(evidence)
                continue
            pathway_id = str(item.get("pathway_id") or "").strip()
            if pathway_id not in self.knowledge.pathway_labels:
                pathway_id = self.pathway_by_label.get(_normalized(str(item.get("term", ""))), "")
            if not pathway_id:
                rejected.append(RejectedFeature("pathway", str(item.get("term", "")), "unmapped pathway", item))
                continue
            weight = self._accept(
                accepted,
                "pathway",
                pathway_id,
                self.knowledge.pathway_labels[pathway_id],
                evidence,
            )
            record["pathways"][pathway_id] = max(weight, record["pathways"].get(pathway_id, 0.0))

        for item in document.get("drugs", [])[:100]:
            if not isinstance(item, dict):
                rejected.append(RejectedFeature("drug", str(item), "item is not an object"))
                continue
            evidence = self._evidence(item, passage_map, "drug")
            if isinstance(evidence, RejectedFeature):
                rejected.append(evidence)
                continue
            role = str(item.get("role", "mentioned"))
            if role in {"mentioned", "comparator"}:
                rejected.append(RejectedFeature("drug", str(item.get("name", "")), f"role is {role}", item))
                continue
            drug = self.knowledge.resolve_drug(str(item.get("chembl_id") or item.get("name") or ""))
            if drug is None:
                rejected.append(RejectedFeature("drug", str(item.get("name", "")), "unmapped drug", item))
                continue
            weight = self._accept(
                accepted,
                "drug",
                drug,
                self.knowledge.drug_labels.get(drug, str(item.get("name", ""))),
                evidence,
                role=role,
            )
            record["drugs"][drug] = max(weight, record["drugs"].get(drug, 0.0))

        for field, allowed in (
            ("inheritance", set(INHERITANCE_MAP)),
            ("onset", set(ONSET_BINS) | {"All ages"}),
        ):
            for item in document.get(field, [])[:20]:
                if not isinstance(item, dict):
                    rejected.append(RejectedFeature(field, str(item), "item is not an object"))
                    continue
                evidence = self._evidence(item, passage_map, field)
                if isinstance(evidence, RejectedFeature):
                    rejected.append(evidence)
                    continue
                value = str(item.get("value", "")).strip()
                if value not in allowed:
                    rejected.append(RejectedFeature(field, value, "value is outside controlled vocabulary", item))
                    continue
                self._accept(accepted, field, value, value, evidence)
                if value not in record[field]:
                    record[field].append(value)

        prevalence_values: list[float] = []
        if record.get("prevalence") is not None:
            prevalence_values.append(float(record["prevalence"]))
        for item in document.get("prevalence", [])[:20]:
            if not isinstance(item, dict):
                rejected.append(RejectedFeature("prevalence", str(item), "item is not an object"))
                continue
            evidence = self._evidence(item, passage_map, "prevalence")
            if isinstance(evidence, RejectedFeature):
                rejected.append(evidence)
                continue
            try:
                value = float(item.get("value_per_person"))
            except (TypeError, ValueError):
                value = math.nan
            if not math.isfinite(value) or not 0 < value <= 1:
                rejected.append(RejectedFeature("prevalence", str(value), "invalid per-person prevalence", item))
                continue
            self._accept(
                accepted,
                "prevalence",
                f"{value:.12g}",
                f"{value:.12g} per person",
                evidence,
                population=str(item.get("population", "")),
            )
            prevalence_values.append(value)
        record["prevalence"] = min(prevalence_values) if prevalence_values else None
        return record, accepted, rejected

    def extract(
        self,
        text: str | None = None,
        *,
        units: Sequence[SourceUnit] | None = None,
        focal_hint: str = "",
        source_id: str = "",
        use_cache: bool = True,
    ) -> ExtractionResult:
        if units is None:
            units = units_from_text(text or "")
        passages = select_passages(units, max_words=self.max_input_words)
        selected_text = "\n\n".join(passage.text for passage in passages)
        baseline, _ = self.deterministic.extract(selected_text, structured=True)
        baseline["description"] = selected_text
        baseline.setdefault("pathways", {})
        source_hash = hashlib.sha256(selected_text.encode("utf-8")).hexdigest()
        warnings: list[str] = []
        completion: Completion | None = None
        document: dict[str, Any] = {}
        status = "deterministic_only"
        if self.client is not None:
            try:
                completion = self.client.complete_json(
                    SYSTEM_PROMPT,
                    self._prompt(passages, focal_hint, baseline),
                    EXTRACTION_SCHEMA,
                    PROMPT_VERSION,
                    use_cache=use_cache,
                )
                document = completion.document
                status = "hybrid_success"
            except MedGemmaError as error:
                if self.strict_llm:
                    raise
                warnings.append(str(error))
                status = "deterministic_fallback"
        record, accepted, rejected = self._ground(
            document,
            passages,
            baseline,
            preserve_baseline_features=status != "hybrid_success",
        )
        if not source_id:
            source_id = source_hash[:16]
        return ExtractionResult(
            record=record,
            focal_disease_label=str(document.get("focal_disease_label", "")).strip(),
            passages=[asdict(passage) for passage in passages],
            accepted_evidence=[asdict(item) for item in accepted],
            rejected_features=[asdict(item) for item in rejected],
            warnings=warnings,
            status=status,
            model=completion.model if completion else "",
            model_revision=completion.model_revision if completion else "",
            prompt_version=PROMPT_VERSION,
            response_sha256=completion.response_sha256 if completion else "",
            cache_key=completion.cache_key if completion else "",
            cached=completion.cached if completion else False,
            usage=completion.usage if completion else {},
            source_id=source_id,
            source_sha256=source_hash,
        )


def save_extraction(path: Path, result: ExtractionResult, provenance: dict[str, Any] | None = None) -> None:
    document = result.to_dict()
    if provenance:
        document["provenance"] = provenance
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)

