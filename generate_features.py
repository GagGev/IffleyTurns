"""Generate disease-level features for rare-disease similarity detection.

The input is the core database bundle produced by ``download_databases.py``.
Outputs are ORPHA-centred Parquet tables under ``.data/features``.  Large inputs
are streamed, and every normalized evidence row retains source information so
that downstream similarity scores can be explained.
"""

from __future__ import annotations

import argparse
import csv
import html.parser
import json
import math
import os
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, Iterator, List, Mapping
from typing import MutableMapping, Optional, Sequence, Set, Tuple

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
except ImportError:  # Keep ``--help`` usable without the optional dependency.
    pa = None
    pc = None
    pq = None


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / ".data" / "databases"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".data" / "features"
SCHEMA_VERSION = "1.0.0"
OUTPUT_NAMES = (
    "diseases.parquet",
    "phenotypes.parquet",
    "genes.parquet",
    "classifications.parquet",
    "prevalence.parquet",
    "treatments.parquet",
    "feature_manifest.json",
)
FREQUENCY_WEIGHTS = {
    "obligate": 1.0,
    "very frequent": 0.895,
    "frequent": 0.545,
    "occasional": 0.17,
    "very rare": 0.025,
    "excluded": 0.0,
}
HPO_FREQUENCY_WEIGHTS = {
    "HP:0040280": 1.0,
    "HP:0040281": 0.895,
    "HP:0040282": 0.545,
    "HP:0040283": 0.17,
    "HP:0040284": 0.025,
    "HP:0040285": 0.0,
}
MOI_NAMES = {
    "AD": "Autosomal dominant",
    "AR": "Autosomal recessive",
    "XL": "X-linked",
    "MT": "Mitochondrial",
    "SD": "Semidominant",
    "UD": "Undetermined",
}


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def clean_text(value: Optional[str]) -> str:
    """Collapse whitespace and return an empty string for missing text."""

    return re.sub(r"\s+", " ", value or "").strip()


def element_text(element: Optional[ET.Element]) -> str:
    """Return all text contained by an XML element."""

    if element is None:
        return ""
    return clean_text("".join(element.itertext()))


def normalize_mondo(value: str) -> str:
    """Normalize MONDO identifiers to ``MONDO:NNNNNNN``."""

    match = re.search(r"(\d+)", value or "")
    return f"MONDO:{match.group(1).zfill(7)}" if match else ""


def normalize_orpha(value: str) -> str:
    """Normalize an Orphanet code to ``ORPHA:N``."""

    match = re.search(r"(\d+)", value or "")
    return f"ORPHA:{int(match.group(1))}" if match else ""


def normalize_hpo(value: str) -> str:
    """Normalize an HPO identifier to its colon form."""

    if not value:
        return ""
    return value.replace("HP_", "HP:")


def normalize_ot_id(value: str) -> str:
    """Normalize a colon identifier to Open Targets' underscore form."""

    return value.replace(":", "_", 1) if ":" in value else value


def normalize_name(value: str) -> str:
    """Normalize a disease name for conservative exact matching."""

    decomposed = unicodedata.normalize("NFKD", value or "")
    ascii_value = "".join(char for char in decomposed if not unicodedata.combining(char))
    return clean_text(re.sub(r"[^a-z0-9]+", " ", ascii_value.lower()))


def designation_disease_name(value: str) -> str:
    """Remove a regulatory indication prefix without fuzzy name rewriting."""

    normalized = normalize_name(value)
    prefixes = (
        r"^(?:for )?(?:the )?treatment (?:of|for|in) ",
        r"^(?:for )?(?:the )?prevention (?:of|for|in) ",
        r"^(?:for )?(?:the )?diagnosis (?:of|for|in) ",
        r"^(?:for )?(?:the )?management (?:of|for|in) ",
        r"^for use (?:in|for) ",
    )
    for pattern in prefixes:
        stripped = re.sub(pattern, "", normalized, count=1)
        if stripped != normalized:
            return stripped
    return normalized


def frequency_weight(value: str) -> Optional[float]:
    """Map an Orphanet/HPO frequency label or fraction to a numeric weight."""

    value = clean_text(value)
    if not value:
        return None
    if value in HPO_FREQUENCY_WEIGHTS:
        return HPO_FREQUENCY_WEIGHTS[value]
    lowered = value.lower()
    for label, weight in FREQUENCY_WEIGHTS.items():
        if lowered.startswith(label):
            return weight
    fraction = re.fullmatch(r"(\d+)\s*/\s*(\d+)", value)
    if fraction and int(fraction.group(2)):
        return int(fraction.group(1)) / int(fraction.group(2))
    percent = re.fullmatch(r"(\d+(?:\.\d+)?)\s*%", value)
    if percent:
        return float(percent.group(1)) / 100.0
    return None


def prevalence_class_estimate(value: str) -> Optional[float]:
    """Estimate a per-person prevalence from an Orphanet class label."""

    normalized = clean_text(value).replace(",", "").replace(" ", "")
    if not normalized or normalized.lower() == "unknown":
        return None
    interval = re.fullmatch(r"(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)/(\d+)", normalized)
    if interval:
        low, high, denominator = map(float, interval.groups())
        return ((low + high) / 2.0) / denominator
    bound = re.fullmatch(r"([<>])(\d+(?:\.\d+)?)/(\d+)", normalized)
    if bound:
        operator, numerator, denominator = bound.groups()
        estimate = float(numerator) / float(denominator)
        return estimate / 2.0 if operator == "<" else estimate
    exact = re.fullmatch(r"(\d+(?:\.\d+)?)/(\d+)", normalized)
    if exact:
        return float(exact.group(1)) / float(exact.group(2))
    return None


def sorted_strings(values: Iterable[str]) -> List[str]:
    """Return non-empty unique strings in deterministic order."""

    return sorted({value for value in values if value})


def relative_posix(path: Path, root: Path) -> str:
    """Return a portable source path relative to the database directory."""

    return path.resolve().relative_to(root.resolve()).as_posix()


@dataclass
class Disease:
    """Identity and aggregate feature sets for one Orphanet entity."""

    code: int
    name: str
    expert_url: str
    description: str
    disorder_type: str
    disorder_group: str
    flags: Set[str] = field(default_factory=set)
    synonyms: Set[str] = field(default_factory=set)
    references: DefaultDict[str, Set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    preferential_parent_ids: Set[str] = field(default_factory=set)
    preferential_parent_names: Set[str] = field(default_factory=set)
    classification_ids: Set[str] = field(default_factory=set)
    classification_names: Set[str] = field(default_factory=set)
    category_ids: Set[str] = field(default_factory=set)
    category_names: Set[str] = field(default_factory=set)
    body_system_ids: Set[str] = field(default_factory=set)
    body_system_names: Set[str] = field(default_factory=set)
    ontology_parent_ids: Set[str] = field(default_factory=set)
    hpo_ids: Set[str] = field(default_factory=set)
    gene_symbols: Set[str] = field(default_factory=set)
    inheritance: Set[str] = field(default_factory=set)
    onset: Set[str] = field(default_factory=set)
    prevalence_regions: Set[str] = field(default_factory=set)
    affected_sexes: Set[str] = field(default_factory=set)
    affected_populations: Set[str] = field(default_factory=set)
    drug_ids: Set[str] = field(default_factory=set)
    approved_drug_ids: Set[str] = field(default_factory=set)
    drug_names: Set[str] = field(default_factory=set)
    treatment_types: Set[str] = field(default_factory=set)
    prevalence_summary: Optional[Dict[str, Any]] = None

    @property
    def orpha_id(self) -> str:
        return f"ORPHA:{self.code}"


@dataclass
class MondoTerm:
    """Subset of a MONDO OBO term needed by this generator."""

    mondo_id: str
    name: str = ""
    synonyms: Set[str] = field(default_factory=set)
    parent_ids: Set[str] = field(default_factory=set)
    orpha_ids: Set[str] = field(default_factory=set)
    efo_ids: Set[str] = field(default_factory=set)
    omim_ids: Set[str] = field(default_factory=set)
    obsolete: bool = False


class SourceCatalog:
    """Resolve local database files to their original download URLs."""

    def __init__(self, input_dir: Path) -> None:
        self.input_dir = input_dir
        self.manifest_path = input_dir / "download_manifest.json"
        self.manifest: Dict[str, Any] = {}
        self.urls: Dict[str, str] = {}
        if self.manifest_path.exists():
            with self.manifest_path.open(encoding="utf-8") as stream:
                self.manifest = json.load(stream)
            for item in self.manifest.get("files", []):
                relative_path = str(item.get("relative_path", "")).replace("\\", "/")
                if relative_path:
                    self.urls[relative_path] = str(item.get("url", ""))

    def describe(self, path: Path) -> Tuple[str, str]:
        relative_path = relative_posix(path, self.input_dir)
        return relative_path, self.urls.get(relative_path, "")


class TableHTMLParser(html.parser.HTMLParser):
    """Parse the simple HTML table distributed with an ``.xls`` suffix by FDA."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: List[List[str]] = []
        self.current_row: Optional[List[str]] = None
        self.cell_parts: Optional[List[str]] = None

    def handle_starttag(
        self, tag: str, attrs: List[Tuple[str, Optional[str]]]
    ) -> None:
        del attrs
        if tag.lower() == "tr":
            self.current_row = []
        elif tag.lower() in {"td", "th"} and self.current_row is not None:
            self.cell_parts = []

    def handle_data(self, data: str) -> None:
        if self.cell_parts is not None:
            self.cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in {"td", "th"} and self.cell_parts is not None:
            assert self.current_row is not None
            self.current_row.append(clean_text("".join(self.cell_parts)))
            self.cell_parts = None
        elif lowered == "tr" and self.current_row is not None:
            if any(self.current_row):
                self.rows.append(self.current_row)
            self.current_row = None


def parse_product1(path: Path) -> Tuple[Dict[str, Disease], Dict[str, Any]]:
    """Parse Orphadata nomenclature, aliases, definitions, flags, and crossrefs."""

    catalog: Dict[str, Disease] = {}
    metadata: Dict[str, Any] = {}
    for event, element in ET.iterparse(path, events=("start", "end")):
        if event == "start" and not metadata:
            metadata = dict(element.attrib)
            continue
        if event != "end" or element.tag != "Disorder":
            continue
        code_text = element.findtext("./OrphaCode")
        if not code_text:
            element.clear()
            continue
        code = int(code_text)
        name = clean_text(element.findtext("./Name"))
        descriptions = []
        for section in element.findall("./SummaryInformationList/SummaryInformation/"
                                       "TextSectionList/TextSection"):
            section_type = clean_text(section.findtext("./TextSectionType/Name"))
            if section_type.lower() == "definition":
                descriptions.append(element_text(section.find("./Contents")))
        disease = Disease(
            code=code,
            name=name,
            expert_url=clean_text(element.findtext("./ExpertLink")),
            description=" ".join(descriptions),
            disorder_type=clean_text(element.findtext("./DisorderType/Name")),
            disorder_group=clean_text(element.findtext("./DisorderGroup/Name")),
        )
        disease.synonyms.update(
            clean_text(item.text) for item in element.findall("./SynonymList/Synonym")
        )
        disease.flags.update(
            clean_text(item.findtext("./Label"))
            for item in element.findall("./DisorderFlagList/DisorderFlag")
            if clean_text(item.findtext("./Label"))
        )
        for external in element.findall("./ExternalReferenceList/ExternalReference"):
            source = clean_text(external.findtext("./Source")).upper()
            reference = clean_text(external.findtext("./Reference"))
            if not source or not reference:
                continue
            if source == "MONDO":
                reference = normalize_mondo(reference)
            elif source == "OMIM":
                reference = f"OMIM:{reference.removeprefix('OMIM:')}"
            elif source == "EFO":
                reference = f"EFO:{reference.removeprefix('EFO:')}"
            disease.references[source].add(reference)
        catalog[disease.orpha_id] = disease
        element.clear()
    return catalog, metadata


def parse_obo_terms(path: Path, prefix: str) -> Iterator[Dict[str, Any]]:
    """Yield minimally parsed OBO term dictionaries."""

    term: Optional[Dict[str, Any]] = None
    with path.open(encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.rstrip("\n")
            if line == "[Term]":
                if term and str(term.get("id", "")).startswith(prefix):
                    yield term
                term = {
                    "synonyms": [],
                    "parents": [],
                    "xref_lines": [],
                    "obsolete": False,
                }
                continue
            if line.startswith("[") and line != "[Term]":
                if term and str(term.get("id", "")).startswith(prefix):
                    yield term
                term = None
                continue
            if term is None:
                continue
            if line.startswith("id: "):
                term["id"] = line[4:].strip()
            elif line.startswith("name: "):
                term["name"] = line[6:].strip()
            elif line.startswith("synonym: "):
                match = re.match(r'synonym: "(.*?)"', line)
                if match:
                    term["synonyms"].append(match.group(1))
            elif line.startswith("is_a: "):
                term["parents"].append(line[6:].split()[0])
            elif line.startswith("xref: ") or "skos:exactMatch" in line:
                term["xref_lines"].append(line)
            elif line == "is_obsolete: true":
                term["obsolete"] = True
    if term and str(term.get("id", "")).startswith(prefix):
        yield term


def parse_mondo(path: Path) -> Dict[str, MondoTerm]:
    """Parse MONDO hierarchy, aliases, and ORPHA/EFO/OMIM cross-references."""

    terms: Dict[str, MondoTerm] = {}
    for raw in parse_obo_terms(path, "MONDO:"):
        mondo_id = normalize_mondo(str(raw["id"]))
        term = MondoTerm(
            mondo_id=mondo_id,
            name=clean_text(str(raw.get("name", ""))),
            synonyms={clean_text(value) for value in raw["synonyms"] if clean_text(value)},
            parent_ids={
                normalize_mondo(value)
                for value in raw["parents"]
                if value.startswith("MONDO:")
            },
            obsolete=bool(raw["obsolete"]),
        )
        for line in raw["xref_lines"]:
            for code in re.findall(r"(?:Orphanet|ORPHA):(\d+)", line, re.I):
                term.orpha_ids.add(normalize_orpha(code))
            for code in re.findall(r"EFO:(\d+)", line, re.I):
                term.efo_ids.add(f"EFO:{code}")
            for code in re.findall(r"OMIM(?:\.PS)?:(\d+)", line, re.I):
                term.omim_ids.add(f"OMIM:{code}")
        if not term.obsolete:
            terms[mondo_id] = term
    return terms


def parse_hpo(path: Path) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """Parse HPO labels and immediate parent identifiers."""

    labels: Dict[str, str] = {}
    parents: Dict[str, List[str]] = {}
    for term in parse_obo_terms(path, "HP:"):
        if term["obsolete"]:
            continue
        hpo_id = str(term["id"])
        labels[hpo_id] = clean_text(str(term.get("name", "")))
        parents[hpo_id] = sorted_strings(term["parents"])
    return labels, parents


def build_crosswalks(
    catalog: Mapping[str, Disease],
    mondo_terms: Mapping[str, MondoTerm],
) -> Tuple[
    DefaultDict[str, Set[str]],
    DefaultDict[str, Set[str]],
    DefaultDict[str, Set[str]],
    DefaultDict[str, Set[str]],
]:
    """Build bidirectional MONDO, OMIM, and EFO to ORPHA crosswalks."""

    orpha_to_mondo: DefaultDict[str, Set[str]] = defaultdict(set)
    mondo_to_orpha: DefaultDict[str, Set[str]] = defaultdict(set)
    omim_to_orpha: DefaultDict[str, Set[str]] = defaultdict(set)
    efo_to_orpha: DefaultDict[str, Set[str]] = defaultdict(set)

    for orpha_id, disease in catalog.items():
        for mondo_id in disease.references.get("MONDO", set()):
            orpha_to_mondo[orpha_id].add(mondo_id)
            mondo_to_orpha[mondo_id].add(orpha_id)
        for omim_id in disease.references.get("OMIM", set()):
            omim_to_orpha[omim_id].add(orpha_id)

    for term in mondo_terms.values():
        for orpha_id in term.orpha_ids:
            if orpha_id not in catalog:
                continue
            orpha_to_mondo[orpha_id].add(term.mondo_id)
            mondo_to_orpha[term.mondo_id].add(orpha_id)
        mapped_orphas = mondo_to_orpha.get(term.mondo_id, set())
        for orpha_id in mapped_orphas:
            for omim_id in term.omim_ids:
                omim_to_orpha[omim_id].add(orpha_id)
            for efo_id in term.efo_ids:
                efo_to_orpha[efo_id].add(orpha_id)
    return orpha_to_mondo, mondo_to_orpha, omim_to_orpha, efo_to_orpha


def source_fields(sources: SourceCatalog, path: Path) -> Dict[str, str]:
    """Return the repeated provenance fields used in normalized rows."""

    source_file, source_url = sources.describe(path)
    return {"source_file": source_file, "source_url": source_url}


def parse_product7(
    path: Path,
    diseases: MutableMapping[str, Disease],
    classifications: List[Dict[str, Any]],
    sources: SourceCatalog,
) -> int:
    """Parse Orphanet preferential-parent categories."""

    count = 0
    provenance = source_fields(sources, path)
    for _, element in ET.iterparse(path, events=("end",)):
        if element.tag != "Disorder":
            continue
        orpha_id = normalize_orpha(element.findtext("./OrphaCode") or "")
        disease = diseases.get(orpha_id)
        if disease is not None:
            for association in element.findall(
                "./DisorderDisorderAssociationList/DisorderDisorderAssociation"
            ):
                association_type = clean_text(
                    association.findtext("./DisorderDisorderAssociationType/Name")
                )
                if association_type != "Preferential parent":
                    continue
                target = association.find("./TargetDisorder")
                if target is None:
                    continue
                parent_id = normalize_orpha(target.findtext("./OrphaCode") or "")
                parent_name = clean_text(target.findtext("./Name"))
                disease.preferential_parent_ids.add(parent_id)
                disease.preferential_parent_names.add(parent_name)
                disease.category_ids.add(parent_id)
                disease.category_names.add(parent_name)
                disease.ontology_parent_ids.add(parent_id)
                classifications.append(
                    {
                        "orpha_id": orpha_id,
                        "source": "orphadata_product7",
                        "classification_id": "ORPHADATA:product7",
                        "classification_name": "Orphanet preferential parent",
                        "system_id": parent_id,
                        "system_name": parent_name,
                        "category_id": parent_id,
                        "category_name": parent_name,
                        "path_ids": [parent_id, orpha_id],
                        "path_names": [parent_name, disease.name],
                        **provenance,
                    }
                )
                count += 1
        element.clear()
    return count


def parse_classifications(
    paths: Sequence[Path],
    diseases: MutableMapping[str, Disease],
    classifications: List[Dict[str, Any]],
    sources: SourceCatalog,
) -> int:
    """Parse every Orphanet specialty tree and retain disease ancestor paths."""

    row_count = 0

    def walk(
        node: ET.Element,
        path_ids: List[str],
        path_names: List[str],
        classification_id: str,
        classification_name: str,
        provenance: Dict[str, str],
    ) -> None:
        nonlocal row_count
        disorder = node.find("./Disorder")
        if disorder is None:
            return
        orpha_id = normalize_orpha(disorder.findtext("./OrphaCode") or "")
        name = clean_text(disorder.findtext("./Name"))
        current_ids = path_ids + [orpha_id]
        current_names = path_names + [name]
        disease = diseases.get(orpha_id)
        if disease is not None:
            ancestor_ids = current_ids[:-1]
            ancestor_names = current_names[:-1]
            system_id = current_ids[0] if current_ids else ""
            system_name = current_names[0] if current_names else ""
            category_id = ancestor_ids[-1] if ancestor_ids else ""
            category_name = ancestor_names[-1] if ancestor_names else ""
            disease.classification_ids.add(classification_id)
            disease.classification_names.add(classification_name)
            disease.body_system_ids.add(system_id)
            disease.body_system_names.add(system_name)
            disease.category_ids.update(ancestor_ids)
            disease.category_names.update(ancestor_names)
            disease.ontology_parent_ids.update(ancestor_ids)
            classifications.append(
                {
                    "orpha_id": orpha_id,
                    "source": "orphadata_product3",
                    "classification_id": classification_id,
                    "classification_name": classification_name,
                    "system_id": system_id,
                    "system_name": system_name,
                    "category_id": category_id,
                    "category_name": category_name,
                    "path_ids": current_ids,
                    "path_names": current_names,
                    **provenance,
                }
            )
            row_count += 1
        children = node.find("./ClassificationNodeChildList")
        if children is not None:
            for child in children.findall("./ClassificationNode"):
                walk(
                    child,
                    current_ids,
                    current_names,
                    classification_id,
                    classification_name,
                    provenance,
                )

    for path in sorted(paths):
        root = ET.parse(path).getroot()
        classification = root.find("./ClassificationList/Classification")
        if classification is None:
            continue
        classification_id = clean_text(classification.findtext("./OrphaNumber"))
        if classification_id:
            classification_id = f"ORPHA:{classification_id}"
        classification_name = clean_text(classification.findtext("./Name"))
        provenance = source_fields(sources, path)
        roots = classification.find("./ClassificationNodeRootList")
        if roots is not None:
            for node in roots.findall("./ClassificationNode"):
                walk(
                    node,
                    [],
                    [],
                    classification_id,
                    classification_name,
                    provenance,
                )
        root.clear()
    return row_count


def parse_product4(
    path: Path,
    diseases: MutableMapping[str, Disease],
    hpo_labels: Mapping[str, str],
    hpo_parents: Mapping[str, List[str]],
    phenotypes: List[Dict[str, Any]],
    sources: SourceCatalog,
) -> int:
    """Parse Orphadata disease-to-HPO annotations."""

    count = 0
    provenance = source_fields(sources, path)
    for _, element in ET.iterparse(path, events=("end",)):
        if element.tag != "HPODisorderSetStatus":
            continue
        disorder = element.find("./Disorder")
        if disorder is None:
            element.clear()
            continue
        orpha_id = normalize_orpha(disorder.findtext("./OrphaCode") or "")
        disease = diseases.get(orpha_id)
        if disease is not None:
            for association in disorder.findall(
                "./HPODisorderAssociationList/HPODisorderAssociation"
            ):
                hpo_id = normalize_hpo(association.findtext("./HPO/HPOId") or "")
                label = clean_text(association.findtext("./HPO/HPOTerm"))
                frequency = clean_text(association.findtext("./HPOFrequency/Name"))
                weight = frequency_weight(frequency)
                present = weight != 0.0
                if present:
                    disease.hpo_ids.add(hpo_id)
                phenotypes.append(
                    {
                        "orpha_id": orpha_id,
                        "hpo_id": hpo_id,
                        "hpo_label": label or hpo_labels.get(hpo_id, ""),
                        "parent_hpo_ids": hpo_parents.get(hpo_id, []),
                        "frequency": frequency,
                        "frequency_weight": weight,
                        "qualifier": "",
                        "present": present,
                        "sex": "",
                        "onset": [],
                        "evidence_type": "Orphadata annotation",
                        "references": [],
                        "source": "orphadata_product4",
                        **provenance,
                    }
                )
                count += 1
        element.clear()
    return count


def parse_product6(
    path: Path,
    diseases: MutableMapping[str, Disease],
    genes: List[Dict[str, Any]],
    sources: SourceCatalog,
) -> int:
    """Parse Orphadata gene-disease evidence."""

    count = 0
    provenance = source_fields(sources, path)
    for _, element in ET.iterparse(path, events=("end",)):
        if element.tag != "Disorder":
            continue
        orpha_id = normalize_orpha(element.findtext("./OrphaCode") or "")
        disease = diseases.get(orpha_id)
        if disease is not None:
            for association in element.findall(
                "./DisorderGeneAssociationList/DisorderGeneAssociation"
            ):
                gene = association.find("./Gene")
                if gene is None:
                    continue
                symbol = clean_text(gene.findtext("./Symbol"))
                refs: DefaultDict[str, Set[str]] = defaultdict(set)
                for external in gene.findall(
                    "./ExternalReferenceList/ExternalReference"
                ):
                    refs[clean_text(external.findtext("./Source")).upper()].add(
                        clean_text(external.findtext("./Reference"))
                    )
                ensembl_id = next(iter(sorted(refs.get("ENSEMBL", set()))), "")
                hgnc_value = next(iter(sorted(refs.get("HGNC", set()))), "")
                hgnc_id = (
                    f"HGNC:{hgnc_value.removeprefix('HGNC:')}" if hgnc_value else ""
                )
                if symbol:
                    disease.gene_symbols.add(symbol)
                genes.append(
                    {
                        "orpha_id": orpha_id,
                        "gene_symbol": symbol,
                        "ensembl_id": ensembl_id,
                        "hgnc_id": hgnc_id,
                        "ncbi_gene_id": "",
                        "association_type": clean_text(
                            association.findtext("./DisorderGeneAssociationType/Name")
                        ),
                        "inheritance": "",
                        "classification": clean_text(
                            association.findtext("./DisorderGeneAssociationStatus/Name")
                        ),
                        "association_score": None,
                        "evidence_count": None,
                        "source_reference": clean_text(
                            association.findtext("./SourceOfValidation")
                        ),
                        "source": "orphadata_product6",
                        **provenance,
                    }
                )
                count += 1
        element.clear()
    return count


def parse_ages(
    path: Path,
    diseases: MutableMapping[str, Disease],
) -> int:
    """Parse disease onset and inheritance categories."""

    count = 0
    for _, element in ET.iterparse(path, events=("end",)):
        if element.tag != "Disorder":
            continue
        disease = diseases.get(normalize_orpha(element.findtext("./OrphaCode") or ""))
        if disease is not None:
            disease.onset.update(
                clean_text(item.findtext("./Name"))
                for item in element.findall(
                    "./AverageAgeOfOnsetList/AverageAgeOfOnset"
                )
            )
            disease.inheritance.update(
                clean_text(item.findtext("./Name"))
                for item in element.findall(
                    "./TypeOfInheritanceList/TypeOfInheritance"
                )
            )
            count += 1
        element.clear()
    return count


def prevalence_priority(row: Mapping[str, Any]) -> Tuple[int, int, int]:
    """Return a sort key identifying the most useful prevalence summary row."""

    prevalence_type = str(row["prevalence_type"]).lower()
    region = str(row["region"]).lower()
    return (
        0 if prevalence_type == "point prevalence" else 1,
        0 if region == "worldwide" else 1,
        0 if row["estimated_per_person"] is not None else 1,
    )


def parse_prevalence(
    path: Path,
    diseases: MutableMapping[str, Disease],
    prevalence_rows: List[Dict[str, Any]],
    sources: SourceCatalog,
) -> int:
    """Parse all Orphadata prevalence evidence and select a disease summary."""

    count = 0
    candidates: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    provenance = source_fields(sources, path)
    for _, element in ET.iterparse(path, events=("end",)):
        if element.tag != "Disorder":
            continue
        orpha_id = normalize_orpha(element.findtext("./OrphaCode") or "")
        disease = diseases.get(orpha_id)
        if disease is not None:
            for prevalence in element.findall("./PrevalenceList/Prevalence"):
                class_name = clean_text(prevalence.findtext("./PrevalenceClass/Name"))
                val_moy_text = clean_text(prevalence.findtext("./ValMoy"))
                try:
                    val_moy = float(val_moy_text) if val_moy_text else None
                except ValueError:
                    val_moy = None
                row = {
                    "orpha_id": orpha_id,
                    "prevalence_type": clean_text(
                        prevalence.findtext("./PrevalenceType/Name")
                    ),
                    "qualification": clean_text(
                        prevalence.findtext("./PrevalenceQualification/Name")
                    ),
                    "prevalence_class": class_name,
                    "val_moy": val_moy,
                    "estimated_per_person": prevalence_class_estimate(class_name),
                    "region": clean_text(
                        prevalence.findtext("./PrevalenceGeographic/Name")
                    ),
                    "validation_status": clean_text(
                        prevalence.findtext("./PrevalenceValidationStatus/Name")
                    ),
                    "source_reference": clean_text(prevalence.findtext("./Source")),
                    "source": "orphadata_product9_prevalence",
                    **provenance,
                }
                prevalence_rows.append(row)
                candidates[orpha_id].append(row)
                disease.prevalence_regions.add(str(row["region"]))
                count += 1
        element.clear()
    for orpha_id, rows in candidates.items():
        diseases[orpha_id].prevalence_summary = min(rows, key=prevalence_priority)
    return count


def add_mondo_features(
    diseases: MutableMapping[str, Disease],
    mondo_terms: Mapping[str, MondoTerm],
    orpha_to_mondo: Mapping[str, Set[str]],
    classifications: List[Dict[str, Any]],
    sources: SourceCatalog,
    path: Path,
) -> int:
    """Attach MONDO aliases, cross-references, and immediate parents."""

    count = 0
    provenance = source_fields(sources, path)
    for orpha_id, mondo_ids in orpha_to_mondo.items():
        disease = diseases.get(orpha_id)
        if disease is None:
            continue
        for mondo_id in mondo_ids:
            term = mondo_terms.get(mondo_id)
            if term is None:
                continue
            disease.synonyms.update(term.synonyms)
            disease.references["MONDO"].add(mondo_id)
            disease.references["EFO"].update(term.efo_ids)
            disease.references["OMIM"].update(term.omim_ids)
            for parent_id in term.parent_ids:
                parent_name = mondo_terms.get(parent_id, MondoTerm(parent_id)).name
                disease.ontology_parent_ids.add(parent_id)
                disease.category_ids.add(parent_id)
                disease.category_names.add(parent_name)
                classifications.append(
                    {
                        "orpha_id": orpha_id,
                        "source": "mondo",
                        "classification_id": mondo_id,
                        "classification_name": term.name,
                        "system_id": "",
                        "system_name": "",
                        "category_id": parent_id,
                        "category_name": parent_name,
                        "path_ids": [parent_id, mondo_id],
                        "path_names": [parent_name, term.name],
                        **provenance,
                    }
                )
                count += 1
    return count


def mapped_orphas_for_external_id(
    value: str,
    diseases: Mapping[str, Disease],
    mondo_to_orpha: Mapping[str, Set[str]],
    efo_to_orpha: Mapping[str, Set[str]],
    omim_to_orpha: Mapping[str, Set[str]],
) -> Set[str]:
    """Resolve a source identifier to included ORPHA records."""

    value = clean_text(value)
    candidates: Set[str] = set()
    if re.match(r"^(?:ORPHA|Orphanet)[:_]", value, re.I):
        candidates.add(normalize_orpha(value))
    elif value.upper().startswith("MONDO"):
        candidates.update(mondo_to_orpha.get(normalize_mondo(value), set()))
    elif value.upper().startswith("EFO"):
        normalized = value.replace("_", ":", 1).upper()
        candidates.update(efo_to_orpha.get(normalized, set()))
    elif value.upper().startswith("OMIM"):
        normalized = value.replace("_", ":", 1).upper()
        candidates.update(omim_to_orpha.get(normalized, set()))
    return {orpha_id for orpha_id in candidates if orpha_id in diseases}


def parse_hpo_genes(
    path: Path,
    diseases: MutableMapping[str, Disease],
    omim_to_orpha: Mapping[str, Set[str]],
    genes: List[Dict[str, Any]],
    sources: SourceCatalog,
) -> int:
    """Parse HPO gene-to-disease annotations."""

    count = 0
    provenance = source_fields(sources, path)
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        for row in reader:
            disease_id = clean_text(row.get("disease_id"))
            if disease_id.startswith("ORPHA:"):
                orpha_ids = {disease_id} if disease_id in diseases else set()
            else:
                orpha_ids = {
                    item
                    for item in omim_to_orpha.get(disease_id, set())
                    if item in diseases
                }
            symbol = clean_text(row.get("gene_symbol"))
            for orpha_id in sorted(orpha_ids):
                diseases[orpha_id].gene_symbols.add(symbol)
                genes.append(
                    {
                        "orpha_id": orpha_id,
                        "gene_symbol": symbol,
                        "ensembl_id": "",
                        "hgnc_id": "",
                        "ncbi_gene_id": clean_text(row.get("ncbi_gene_id")),
                        "association_type": clean_text(row.get("association_type")),
                        "inheritance": "",
                        "classification": "",
                        "association_score": None,
                        "evidence_count": None,
                        "source_reference": clean_text(row.get("source")),
                        "source": "hpo_genes_to_disease",
                        **provenance,
                    }
                )
                count += 1
    return count


def parse_clingen(
    path: Path,
    diseases: MutableMapping[str, Disease],
    mondo_to_orpha: Mapping[str, Set[str]],
    genes: List[Dict[str, Any]],
    sources: SourceCatalog,
) -> int:
    """Parse ClinGen curated gene-disease validity assertions."""

    count = 0
    provenance = source_fields(sources, path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.reader(stream)
        header: Optional[List[str]] = None
        for values in rows:
            if values and values[0] == "GENE SYMBOL":
                header = values
                break
        if header is None:
            raise ValueError(f"ClinGen header not found in {path}")
        for values in rows:
            if not values or values[0].startswith("+++"):
                continue
            padded = values + [""] * (len(header) - len(values))
            row = dict(zip(header, padded))
            mondo_id = normalize_mondo(row.get("DISEASE ID (MONDO)", ""))
            orpha_ids = {
                item for item in mondo_to_orpha.get(mondo_id, set()) if item in diseases
            }
            symbol = clean_text(row.get("GENE SYMBOL"))
            moi = clean_text(row.get("MOI"))
            inheritance = MOI_NAMES.get(moi, moi)
            for orpha_id in sorted(orpha_ids):
                diseases[orpha_id].gene_symbols.add(symbol)
                if inheritance:
                    diseases[orpha_id].inheritance.add(inheritance)
                genes.append(
                    {
                        "orpha_id": orpha_id,
                        "gene_symbol": symbol,
                        "ensembl_id": "",
                        "hgnc_id": clean_text(row.get("GENE ID (HGNC)")),
                        "ncbi_gene_id": "",
                        "association_type": "ClinGen gene-disease validity",
                        "inheritance": inheritance,
                        "classification": clean_text(row.get("CLASSIFICATION")),
                        "association_score": None,
                        "evidence_count": None,
                        "source_reference": clean_text(row.get("ONLINE REPORT")),
                        "source": "clingen_gene_validity",
                        **provenance,
                    }
                )
                count += 1
    return count


def parse_clinvar_gene_conditions(
    path: Path,
    diseases: MutableMapping[str, Disease],
    mondo_to_orpha: Mapping[str, Set[str]],
    omim_to_orpha: Mapping[str, Set[str]],
    genes: List[Dict[str, Any]],
    sources: SourceCatalog,
) -> int:
    """Parse ClinVar's compact gene-condition cross-reference table."""

    count = 0
    provenance = source_fields(sources, path)
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        for row in reader:
            source_id = clean_text(row.get("SourceID"))
            disease_mim = clean_text(row.get("DiseaseMIM"))
            if source_id.startswith("MONDO:"):
                orpha_ids = mondo_to_orpha.get(normalize_mondo(source_id), set())
            elif disease_mim:
                orpha_ids = omim_to_orpha.get(f"OMIM:{disease_mim}", set())
            else:
                orpha_ids = set()
            symbols = [
                clean_text(value)
                for value in re.split(r"[|,;]", clean_text(row.get("AssociatedGenes")))
                if clean_text(value)
            ]
            for orpha_id in sorted(set(orpha_ids) & set(diseases)):
                for symbol in symbols:
                    diseases[orpha_id].gene_symbols.add(symbol)
                    genes.append(
                        {
                            "orpha_id": orpha_id,
                            "gene_symbol": symbol,
                            "ensembl_id": "",
                            "hgnc_id": "",
                            "ncbi_gene_id": (
                                f"NCBIGene:{clean_text(row.get('#GeneID'))}"
                                if clean_text(row.get("#GeneID"))
                                else ""
                            ),
                            "association_type": "ClinVar gene-condition",
                            "inheritance": "",
                            "classification": "",
                            "association_score": None,
                            "evidence_count": None,
                            "source_reference": source_id or disease_mim,
                            "source": "clinvar_gene_condition",
                            **provenance,
                        }
                    )
                    count += 1
    return count


def require_pyarrow() -> None:
    """Fail with an actionable message when Parquet support is unavailable."""

    if pa is None or pc is None or pq is None:
        raise RuntimeError(
            "pyarrow is required to generate features. Install it with "
            "`python -m pip install pyarrow`."
        )


def read_parquet_rows(path: Path, columns: Sequence[str]) -> List[Dict[str, Any]]:
    """Read selected columns from a small Parquet input."""

    assert pq is not None
    return pq.read_table(path, columns=list(columns)).to_pylist()


def parse_open_targets_diseases(
    path: Path,
    diseases: MutableMapping[str, Disease],
    mondo_to_orpha: Mapping[str, Set[str]],
    efo_to_orpha: Mapping[str, Set[str]],
    omim_to_orpha: Mapping[str, Set[str]],
    classifications: List[Dict[str, Any]],
    sources: SourceCatalog,
) -> Tuple[Dict[str, Set[str]], Dict[str, str], int]:
    """Build the Open Targets disease crosswalk and ontology features."""

    rows = read_parquet_rows(
        path,
        (
            "id",
            "name",
            "dbXRefs",
            "parents",
            "ancestors",
            "therapeuticAreas",
            "exactSynonyms",
        ),
    )
    names = {str(row["id"]): clean_text(row.get("name")) for row in rows}
    ot_to_orpha: Dict[str, Set[str]] = {}
    for row in rows:
        external_id = str(row["id"])
        matched = mapped_orphas_for_external_id(
            external_id,
            diseases,
            mondo_to_orpha,
            efo_to_orpha,
            omim_to_orpha,
        )
        for crossref in row.get("dbXRefs") or []:
            matched.update(
                mapped_orphas_for_external_id(
                    str(crossref),
                    diseases,
                    mondo_to_orpha,
                    efo_to_orpha,
                    omim_to_orpha,
                )
            )
        if matched:
            ot_to_orpha[external_id] = matched

    provenance = source_fields(sources, path)
    count = 0
    for row in rows:
        external_id = str(row["id"])
        orpha_ids = ot_to_orpha.get(external_id, set())
        if not orpha_ids:
            continue
        parents = [str(value) for value in row.get("parents") or []]
        ancestors = [str(value) for value in row.get("ancestors") or []]
        therapeutic_areas = [
            str(value) for value in row.get("therapeuticAreas") or []
        ]
        for orpha_id in sorted(orpha_ids):
            disease = diseases[orpha_id]
            disease.synonyms.update(
                clean_text(value) for value in row.get("exactSynonyms") or []
            )
            disease.ontology_parent_ids.update(parents)
            disease.ontology_parent_ids.update(ancestors)
            disease.category_ids.update(parents)
            disease.category_names.update(names.get(value, "") for value in parents)
            disease.body_system_ids.update(therapeutic_areas)
            disease.body_system_names.update(
                names.get(value, "") for value in therapeutic_areas
            )
            for parent_id in parents:
                classifications.append(
                    {
                        "orpha_id": orpha_id,
                        "source": "opentargets",
                        "classification_id": external_id,
                        "classification_name": clean_text(row.get("name")),
                        "system_id": (
                            therapeutic_areas[0] if len(therapeutic_areas) == 1 else ""
                        ),
                        "system_name": (
                            names.get(therapeutic_areas[0], "")
                            if len(therapeutic_areas) == 1
                            else ""
                        ),
                        "category_id": parent_id,
                        "category_name": names.get(parent_id, ""),
                        "path_ids": [parent_id, external_id],
                        "path_names": [
                            names.get(parent_id, ""),
                            clean_text(row.get("name")),
                        ],
                        **provenance,
                    }
                )
                count += 1
    return ot_to_orpha, names, count


def parse_open_targets_phenotypes(
    path: Path,
    diseases: MutableMapping[str, Disease],
    ot_to_orpha: Mapping[str, Set[str]],
    hpo_labels: Mapping[str, str],
    hpo_parents: Mapping[str, List[str]],
    phenotypes: List[Dict[str, Any]],
    sources: SourceCatalog,
) -> int:
    """Parse Open Targets phenotype evidence, including sparse sex and onset."""

    count = 0
    provenance = source_fields(sources, path)
    rows = read_parquet_rows(path, ("disease", "phenotype", "evidence"))
    for row in rows:
        orpha_ids = ot_to_orpha.get(str(row["disease"]), set())
        if not orpha_ids:
            continue
        hpo_id = normalize_hpo(str(row["phenotype"]))
        evidence_items = row.get("evidence") or [{}]
        for evidence in evidence_items:
            frequency = clean_text(evidence.get("frequency"))
            sex = clean_text(evidence.get("sex")).upper()
            onset = sorted_strings(
                normalize_hpo(str(value)) for value in evidence.get("onset") or []
            )
            present = not bool(evidence.get("qualifierNot"))
            for orpha_id in sorted(orpha_ids):
                disease = diseases[orpha_id]
                if present:
                    disease.hpo_ids.add(hpo_id)
                if sex:
                    disease.affected_sexes.add(sex)
                phenotypes.append(
                    {
                        "orpha_id": orpha_id,
                        "hpo_id": hpo_id,
                        "hpo_label": hpo_labels.get(hpo_id, ""),
                        "parent_hpo_ids": hpo_parents.get(hpo_id, []),
                        "frequency": frequency,
                        "frequency_weight": frequency_weight(frequency),
                        "qualifier": clean_text(evidence.get("qualifier")),
                        "present": present,
                        "sex": sex,
                        "onset": onset,
                        "evidence_type": clean_text(evidence.get("evidenceType")),
                        "references": sorted_strings(
                            str(value) for value in evidence.get("references") or []
                        ),
                        "source": "opentargets_disease_phenotype",
                        **provenance,
                    }
                )
                count += 1
    return count


def parse_open_targets_targets(paths: Sequence[Path]) -> Dict[str, Dict[str, str]]:
    """Map Ensembl target IDs to symbols and HGNC identifiers."""

    targets: Dict[str, Dict[str, str]] = {}
    for path in sorted(paths):
        for row in read_parquet_rows(path, ("id", "approvedSymbol", "dbXrefs")):
            hgnc_id = ""
            for crossref in row.get("dbXrefs") or []:
                if str(crossref.get("source", "")).upper() == "HGNC":
                    value = clean_text(crossref.get("id"))
                    hgnc_id = (
                        value if value.startswith("HGNC:") else f"HGNC:{value}"
                    )
                    break
            targets[str(row["id"])] = {
                "symbol": clean_text(row.get("approvedSymbol")),
                "hgnc_id": hgnc_id,
            }
    return targets


def parse_open_targets_drugs(path: Path) -> Dict[str, Dict[str, Any]]:
    """Read the compact Open Targets drug dictionary."""

    return {
        str(row["id"]): row
        for row in read_parquet_rows(
            path,
            (
                "id",
                "name",
                "drugType",
                "maximumClinicalStage",
                "synonyms",
                "crossReferences",
            ),
        )
    }


def parse_open_targets_treatments(
    path: Path,
    diseases: MutableMapping[str, Disease],
    ot_to_orpha: Mapping[str, Set[str]],
    drugs: Mapping[str, Dict[str, Any]],
    treatments: List[Dict[str, Any]],
    sources: SourceCatalog,
) -> int:
    """Parse Open Targets drug indications and approval stages."""

    count = 0
    provenance = source_fields(sources, path)
    for row in read_parquet_rows(
        path, ("id", "maxClinicalStage", "diseaseId", "drugId")
    ):
        orpha_ids = ot_to_orpha.get(str(row["diseaseId"]), set())
        if not orpha_ids:
            continue
        drug_id = str(row["drugId"])
        drug = drugs.get(drug_id, {})
        stage = clean_text(row.get("maxClinicalStage"))
        approved = stage.upper() == "APPROVAL"
        name = clean_text(drug.get("name"))
        treatment_type = clean_text(drug.get("drugType"))
        for orpha_id in sorted(orpha_ids):
            disease = diseases[orpha_id]
            disease.drug_ids.add(drug_id)
            disease.drug_names.add(name)
            disease.treatment_types.add(treatment_type)
            if approved:
                disease.approved_drug_ids.add(drug_id)
            treatments.append(
                {
                    "orpha_id": orpha_id,
                    "drug_id": drug_id,
                    "drug_name": name,
                    "treatment_type": treatment_type,
                    "max_clinical_stage": stage,
                    "is_approved": approved,
                    "indication": "",
                    "status": "",
                    "designation_id": clean_text(row.get("id")),
                    "exact_name_match": False,
                    "source": "opentargets_clinical_indication",
                    **provenance,
                }
            )
            count += 1
    return count


def build_name_index(diseases: Mapping[str, Disease]) -> Dict[str, Set[str]]:
    """Build a normalized disease-name and alias index."""

    index: DefaultDict[str, Set[str]] = defaultdict(set)
    for orpha_id, disease in diseases.items():
        for name in {disease.name, *disease.synonyms}:
            normalized = normalize_name(name)
            if normalized:
                index[normalized].add(orpha_id)
    return dict(index)


def parse_ema_treatments(
    path: Path,
    diseases: MutableMapping[str, Disease],
    name_index: Mapping[str, Set[str]],
    treatments: List[Dict[str, Any]],
    sources: SourceCatalog,
) -> Tuple[int, int, int]:
    """Attach EMA orphan designations through unique exact disease-name matches."""

    with path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    matched = 0
    unmatched = 0
    ambiguous = 0
    provenance = source_fields(sources, path)
    for row in document.get("data", []):
        indication = clean_text(row.get("intended_use"))
        matches = name_index.get(designation_disease_name(indication), set())
        if not matches:
            unmatched += 1
            continue
        if len(matches) != 1:
            ambiguous += 1
            continue
        orpha_id = next(iter(matches))
        drug_name = clean_text(row.get("active_substance")) or clean_text(
            row.get("medicine_name")
        )
        designation_id = clean_text(row.get("eu_designation_number"))
        drug_id = f"EMA:{designation_id}" if designation_id else f"EMA:{drug_name}"
        disease = diseases[orpha_id]
        disease.drug_ids.add(drug_id)
        disease.drug_names.add(drug_name)
        disease.treatment_types.add("Orphan designation")
        treatments.append(
            {
                "orpha_id": orpha_id,
                "drug_id": drug_id,
                "drug_name": drug_name,
                "treatment_type": "Orphan designation",
                "max_clinical_stage": "",
                "is_approved": None,
                "indication": indication,
                "status": clean_text(row.get("status")),
                "designation_id": designation_id,
                "exact_name_match": True,
                "source": "ema_orphan_designation",
                "source_file": provenance["source_file"],
                "source_url": clean_text(row.get("orphan_designation_url"))
                or provenance["source_url"],
            }
        )
        matched += 1
    return matched, unmatched, ambiguous


def parse_fda_treatments(
    path: Path,
    diseases: MutableMapping[str, Disease],
    name_index: Mapping[str, Set[str]],
    treatments: List[Dict[str, Any]],
    sources: SourceCatalog,
) -> Tuple[int, int, int]:
    """Attach FDA orphan designations through unique exact disease-name matches."""

    parser = TableHTMLParser()
    with path.open(encoding="utf-8", errors="replace") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), ""):
            parser.feed(chunk)
    parser.close()
    if not parser.rows:
        raise ValueError(f"No FDA table rows found in {path}")
    headers = parser.rows[0]
    matched = 0
    unmatched = 0
    ambiguous = 0
    provenance = source_fields(sources, path)
    for values in parser.rows[1:]:
        padded = values + [""] * (len(headers) - len(values))
        row = dict(zip(headers, padded))
        indication = clean_text(row.get("Orphan Designation"))
        matches = name_index.get(designation_disease_name(indication), set())
        if not matches:
            unmatched += 1
            continue
        if len(matches) != 1:
            ambiguous += 1
            continue
        orpha_id = next(iter(matches))
        drug_name = clean_text(row.get("Generic Name"))
        status = clean_text(row.get("FDA Orphan Approval Status"))
        approved = status.lower().startswith("approved")
        designation_status = clean_text(row.get("Orphan Designation Status"))
        designation_id = clean_text(row.get("Designation Number"))
        stable_part = designation_id or f"{drug_name}:{indication}"
        drug_id = f"FDA:{stable_part}"
        disease = diseases[orpha_id]
        disease.drug_ids.add(drug_id)
        disease.drug_names.add(drug_name)
        disease.treatment_types.add("Orphan designation")
        if approved:
            disease.approved_drug_ids.add(drug_id)
        treatments.append(
            {
                "orpha_id": orpha_id,
                "drug_id": drug_id,
                "drug_name": drug_name,
                "treatment_type": "Orphan designation",
                "max_clinical_stage": "APPROVAL" if approved else "",
                "is_approved": approved,
                "indication": indication,
                "status": designation_status or status,
                "designation_id": designation_id,
                "exact_name_match": True,
                "source": "fda_orphan_designation",
                **provenance,
            }
        )
        matched += 1
    return matched, unmatched, ambiguous


def schemas() -> Dict[str, Any]:
    """Return explicit stable output schemas."""

    assert pa is not None
    strings = pa.list_(pa.string())
    return {
        "diseases.parquet": pa.schema(
            [
                ("orpha_id", pa.string()),
                ("orpha_code", pa.int64()),
                ("name", pa.string()),
                ("synonyms", strings),
                ("expert_url", pa.string()),
                ("description", pa.string()),
                ("disorder_type", pa.string()),
                ("disorder_group", pa.string()),
                ("flags", strings),
                ("mondo_ids", strings),
                ("omim_ids", strings),
                ("efo_ids", strings),
                ("icd10_ids", strings),
                ("icd11_ids", strings),
                ("umls_ids", strings),
                ("gard_ids", strings),
                ("mesh_ids", strings),
                ("preferential_parent_ids", strings),
                ("preferential_parent_names", strings),
                ("classification_ids", strings),
                ("classification_names", strings),
                ("category_ids", strings),
                ("category_names", strings),
                ("body_system_ids", strings),
                ("body_system_names", strings),
                ("ontology_parent_ids", strings),
                ("hpo_ids", strings),
                ("gene_symbols", strings),
                ("inheritance", strings),
                ("onset", strings),
                ("prevalence_type", pa.string()),
                ("prevalence_class", pa.string()),
                ("prevalence_region", pa.string()),
                ("prevalence_estimated_per_person", pa.float64()),
                ("prevalence_regions", strings),
                ("affected_sexes", strings),
                ("affected_populations", strings),
                ("drug_ids", strings),
                ("approved_drug_ids", strings),
                ("drug_names", strings),
                ("treatment_types", strings),
                ("n_hpo_phenotypes", pa.int64()),
                ("n_genes", pa.int64()),
                ("n_drugs", pa.int64()),
                ("n_approved_drugs", pa.int64()),
            ]
        ),
        "phenotypes.parquet": pa.schema(
            [
                ("orpha_id", pa.string()),
                ("hpo_id", pa.string()),
                ("hpo_label", pa.string()),
                ("parent_hpo_ids", strings),
                ("frequency", pa.string()),
                ("frequency_weight", pa.float64()),
                ("qualifier", pa.string()),
                ("present", pa.bool_()),
                ("sex", pa.string()),
                ("onset", strings),
                ("evidence_type", pa.string()),
                ("references", strings),
                ("source", pa.string()),
                ("source_file", pa.string()),
                ("source_url", pa.string()),
            ]
        ),
        "genes.parquet": pa.schema(
            [
                ("orpha_id", pa.string()),
                ("gene_symbol", pa.string()),
                ("ensembl_id", pa.string()),
                ("hgnc_id", pa.string()),
                ("ncbi_gene_id", pa.string()),
                ("association_type", pa.string()),
                ("inheritance", pa.string()),
                ("classification", pa.string()),
                ("association_score", pa.float64()),
                ("evidence_count", pa.int64()),
                ("source_reference", pa.string()),
                ("source", pa.string()),
                ("source_file", pa.string()),
                ("source_url", pa.string()),
            ]
        ),
        "classifications.parquet": pa.schema(
            [
                ("orpha_id", pa.string()),
                ("source", pa.string()),
                ("classification_id", pa.string()),
                ("classification_name", pa.string()),
                ("system_id", pa.string()),
                ("system_name", pa.string()),
                ("category_id", pa.string()),
                ("category_name", pa.string()),
                ("path_ids", strings),
                ("path_names", strings),
                ("source_file", pa.string()),
                ("source_url", pa.string()),
            ]
        ),
        "prevalence.parquet": pa.schema(
            [
                ("orpha_id", pa.string()),
                ("prevalence_type", pa.string()),
                ("qualification", pa.string()),
                ("prevalence_class", pa.string()),
                ("val_moy", pa.float64()),
                ("estimated_per_person", pa.float64()),
                ("region", pa.string()),
                ("validation_status", pa.string()),
                ("source_reference", pa.string()),
                ("source", pa.string()),
                ("source_file", pa.string()),
                ("source_url", pa.string()),
            ]
        ),
        "treatments.parquet": pa.schema(
            [
                ("orpha_id", pa.string()),
                ("drug_id", pa.string()),
                ("drug_name", pa.string()),
                ("treatment_type", pa.string()),
                ("max_clinical_stage", pa.string()),
                ("is_approved", pa.bool_()),
                ("indication", pa.string()),
                ("status", pa.string()),
                ("designation_id", pa.string()),
                ("exact_name_match", pa.bool_()),
                ("source", pa.string()),
                ("source_file", pa.string()),
                ("source_url", pa.string()),
            ]
        ),
    }


def table_from_rows(rows: Sequence[Dict[str, Any]], schema: Any) -> Any:
    """Build an Arrow table while enforcing a stable schema for empty/null columns."""

    assert pa is not None
    normalized = [
        {field.name: row.get(field.name) for field in schema}
        for row in rows
    ]
    return pa.Table.from_pylist(normalized, schema=schema)


def write_rows_atomic(
    output_path: Path,
    rows: Sequence[Dict[str, Any]],
    schema: Any,
) -> None:
    """Write a Parquet table to a temporary path and atomically replace it."""

    assert pq is not None
    temporary = output_path.with_name(f"{output_path.name}.part")
    table = table_from_rows(rows, schema)
    pq.write_table(table, temporary, compression="zstd", use_dictionary=True)
    os.replace(temporary, output_path)


def deduplicate_rows(
    rows: Sequence[Dict[str, Any]], fields: Sequence[str]
) -> List[Dict[str, Any]]:
    """Deduplicate normalized rows by scalar identifying fields."""

    seen: Set[Tuple[Any, ...]] = set()
    result: List[Dict[str, Any]] = []
    for row in rows:
        key = tuple(row.get(field) for field in fields)
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def write_genes_with_open_targets(
    output_path: Path,
    base_rows: Sequence[Dict[str, Any]],
    association_paths: Sequence[Path],
    diseases: MutableMapping[str, Disease],
    ot_to_orpha: Mapping[str, Set[str]],
    targets: Mapping[str, Dict[str, str]],
    min_score: float,
    schema: Any,
    sources: SourceCatalog,
) -> Tuple[int, int]:
    """Write gene evidence, streaming filtered Open Targets association batches."""

    assert pa is not None and pc is not None and pq is not None
    temporary = output_path.with_name(f"{output_path.name}.part")
    writer = pq.ParquetWriter(
        temporary, schema, compression="zstd", use_dictionary=True
    )
    base_count = 0
    association_count = 0
    try:
        sorted_base = sorted(
            base_rows,
            key=lambda row: (
                row["orpha_id"],
                row["gene_symbol"],
                row["source"],
                row["source_reference"],
            ),
        )
        if sorted_base:
            writer.write_table(table_from_rows(sorted_base, schema))
            base_count = len(sorted_base)
        mapped_ids = sorted(ot_to_orpha)
        if mapped_ids:
            value_set = pa.array(mapped_ids, type=pa.string())
            for path in sorted(association_paths):
                provenance = source_fields(sources, path)
                parquet_file = pq.ParquetFile(path)
                for batch in parquet_file.iter_batches(
                    batch_size=65536,
                    columns=(
                        "diseaseId",
                        "targetId",
                        "aggregationType",
                        "associationScore",
                        "evidenceCount",
                    ),
                ):
                    mask = pc.is_in(batch.column("diseaseId"), value_set=value_set)
                    filtered = batch.filter(mask)
                    if filtered.num_rows == 0:
                        continue
                    expanded: List[Dict[str, Any]] = []
                    for row in filtered.to_pylist():
                        target_id = str(row["targetId"])
                        target = targets.get(target_id, {})
                        symbol = target.get("symbol", "")
                        score = row.get("associationScore")
                        for orpha_id in sorted(
                            ot_to_orpha.get(str(row["diseaseId"]), set())
                        ):
                            if score is not None and float(score) >= min_score and symbol:
                                diseases[orpha_id].gene_symbols.add(symbol)
                            expanded.append(
                                {
                                    "orpha_id": orpha_id,
                                    "gene_symbol": symbol,
                                    "ensembl_id": target_id,
                                    "hgnc_id": target.get("hgnc_id", ""),
                                    "ncbi_gene_id": "",
                                    "association_type": clean_text(
                                        row.get("aggregationType")
                                    ),
                                    "inheritance": "",
                                    "classification": "",
                                    "association_score": (
                                        float(score) if score is not None else None
                                    ),
                                    "evidence_count": row.get("evidenceCount"),
                                    "source_reference": str(row["diseaseId"]),
                                    "source": "opentargets_association_overall_direct",
                                    **provenance,
                                }
                            )
                    expanded.sort(
                        key=lambda row: (
                            row["orpha_id"],
                            row["gene_symbol"],
                            row["ensembl_id"],
                        )
                    )
                    writer.write_table(table_from_rows(expanded, schema))
                    association_count += len(expanded)
    except BaseException:
        writer.close()
        temporary.unlink(missing_ok=True)
        raise
    else:
        writer.close()
        os.replace(temporary, output_path)
    return base_count, association_count


def disease_rows(diseases: Mapping[str, Disease]) -> List[Dict[str, Any]]:
    """Convert disease aggregates to stable output rows."""

    rows = []
    for disease in diseases.values():
        summary = disease.prevalence_summary or {}
        rows.append(
            {
                "orpha_id": disease.orpha_id,
                "orpha_code": disease.code,
                "name": disease.name,
                "synonyms": sorted_strings(disease.synonyms),
                "expert_url": disease.expert_url,
                "description": disease.description,
                "disorder_type": disease.disorder_type,
                "disorder_group": disease.disorder_group,
                "flags": sorted_strings(disease.flags),
                "mondo_ids": sorted_strings(disease.references.get("MONDO", set())),
                "omim_ids": sorted_strings(disease.references.get("OMIM", set())),
                "efo_ids": sorted_strings(disease.references.get("EFO", set())),
                "icd10_ids": sorted_strings(disease.references.get("ICD-10", set())),
                "icd11_ids": sorted_strings(disease.references.get("ICD-11", set())),
                "umls_ids": sorted_strings(disease.references.get("UMLS", set())),
                "gard_ids": sorted_strings(disease.references.get("GARD", set())),
                "mesh_ids": sorted_strings(disease.references.get("MESH", set())),
                "preferential_parent_ids": sorted_strings(
                    disease.preferential_parent_ids
                ),
                "preferential_parent_names": sorted_strings(
                    disease.preferential_parent_names
                ),
                "classification_ids": sorted_strings(disease.classification_ids),
                "classification_names": sorted_strings(
                    disease.classification_names
                ),
                "category_ids": sorted_strings(disease.category_ids),
                "category_names": sorted_strings(disease.category_names),
                "body_system_ids": sorted_strings(disease.body_system_ids),
                "body_system_names": sorted_strings(disease.body_system_names),
                "ontology_parent_ids": sorted_strings(disease.ontology_parent_ids),
                "hpo_ids": sorted_strings(disease.hpo_ids),
                "gene_symbols": sorted_strings(disease.gene_symbols),
                "inheritance": sorted_strings(disease.inheritance),
                "onset": sorted_strings(disease.onset),
                "prevalence_type": clean_text(summary.get("prevalence_type")),
                "prevalence_class": clean_text(summary.get("prevalence_class")),
                "prevalence_region": clean_text(summary.get("region")),
                "prevalence_estimated_per_person": summary.get(
                    "estimated_per_person"
                ),
                "prevalence_regions": sorted_strings(disease.prevalence_regions),
                "affected_sexes": sorted_strings(disease.affected_sexes),
                "affected_populations": sorted_strings(
                    disease.affected_populations
                ),
                "drug_ids": sorted_strings(disease.drug_ids),
                "approved_drug_ids": sorted_strings(disease.approved_drug_ids),
                "drug_names": sorted_strings(disease.drug_names),
                "treatment_types": sorted_strings(disease.treatment_types),
                "n_hpo_phenotypes": len(disease.hpo_ids),
                "n_genes": len(disease.gene_symbols),
                "n_drugs": len(disease.drug_ids),
                "n_approved_drugs": len(disease.approved_drug_ids),
            }
        )
    return sorted(rows, key=lambda row: int(row["orpha_code"]))


def coverage(diseases: Mapping[str, Disease]) -> Dict[str, int]:
    """Summarize disease-level feature coverage."""

    return {
        "with_mondo_id": sum(
            bool(item.references.get("MONDO")) for item in diseases.values()
        ),
        "with_classification": sum(
            bool(item.category_ids) for item in diseases.values()
        ),
        "with_body_system": sum(
            bool(item.body_system_ids) for item in diseases.values()
        ),
        "with_phenotypes": sum(bool(item.hpo_ids) for item in diseases.values()),
        "with_genes": sum(bool(item.gene_symbols) for item in diseases.values()),
        "with_inheritance": sum(
            bool(item.inheritance) for item in diseases.values()
        ),
        "with_onset": sum(bool(item.onset) for item in diseases.values()),
        "with_prevalence": sum(
            item.prevalence_summary is not None for item in diseases.values()
        ),
        "with_sex_evidence": sum(
            bool(item.affected_sexes) for item in diseases.values()
        ),
        "with_population_evidence": sum(
            bool(item.affected_populations) for item in diseases.values()
        ),
        "with_treatments": sum(bool(item.drug_ids) for item in diseases.values()),
        "with_approved_drugs": sum(
            bool(item.approved_drug_ids) for item in diseases.values()
        ),
    }


def write_json_atomic(path: Path, document: Mapping[str, Any]) -> None:
    """Write JSON atomically."""

    temporary = path.with_name(f"{path.name}.part")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def validate_inputs(input_dir: Path) -> Dict[str, Any]:
    """Resolve required database files and fail before doing expensive work."""

    files: Dict[str, Any] = {
        "product1": input_dir / "orphadata" / "en_product1.xml",
        "product4": input_dir / "orphadata" / "en_product4.xml",
        "product6": input_dir / "orphadata" / "en_product6.xml",
        "product7": input_dir / "orphadata" / "en_product7.xml",
        "ages": input_dir / "orphadata" / "en_product9_ages.xml",
        "prevalence": input_dir / "orphadata" / "en_product9_prev.xml",
        "mondo": input_dir / "mondo" / "mondo.obo",
        "hpo": input_dir / "hpo" / "hp.obo",
        "hpo_genes": input_dir / "hpo" / "genes_to_disease.txt",
        "clingen": input_dir / "clingen" / "gene_disease_validity.csv",
        "clinvar_genes": input_dir / "clinvar" / "gene_condition_source_id",
        "ot_disease": input_dir / "opentargets" / "disease" / "00000000.parquet",
        "ot_phenotype": (
            input_dir / "opentargets" / "disease_phenotype" / "00000000.parquet"
        ),
        "ot_drugs": (
            input_dir / "opentargets" / "drug_molecule" / "00000000.parquet"
        ),
        "ot_indications": (
            input_dir / "opentargets" / "clinical_indication" / "00000000.parquet"
        ),
        "ema": input_dir / "regulatory" / "ema_orphan_designations.json",
        "fda": input_dir / "regulatory" / "fda_orphan_designations.xls",
    }
    files["classifications"] = sorted(
        (input_dir / "orphadata" / "classifications").glob("en_product3_*.xml")
    )
    files["ot_targets"] = sorted(
        (input_dir / "opentargets" / "target").glob("*.parquet")
    )
    files["ot_associations"] = sorted(
        (input_dir / "opentargets" / "association_overall_direct").glob(
            "*.parquet"
        )
    )
    missing = [
        str(path)
        for key, path in files.items()
        if key not in {"classifications", "ot_targets", "ot_associations"}
        and not path.is_file()
    ]
    for key in ("classifications", "ot_targets", "ot_associations"):
        if not files[key]:
            missing.append(f"{key}: no matching files")
    if missing:
        details = "\n  - ".join(missing)
        raise FileNotFoundError(
            f"Required downloaded database inputs are missing:\n  - {details}\n"
            "Run download_databases.py before generating features."
        )
    return files


def check_output_guard(output_dir: Path, force: bool) -> None:
    """Refuse to overwrite a completed feature build unless requested."""

    existing = [output_dir / name for name in OUTPUT_NAMES if (output_dir / name).exists()]
    if existing and not force:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Feature outputs already exist ({names}). Re-run with --force to replace them."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for part in output_dir.glob("*.part"):
        part.unlink(missing_ok=True)


def bounded_score(value: str) -> float:
    """Argparse type for an association score between zero and one."""

    try:
        score = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not 0.0 <= score <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return score


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate ORPHA-centred disease feature and evidence Parquet tables "
            "from the databases downloaded by download_databases.py."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"downloaded database directory (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"feature output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--min-association-score",
        type=bounded_score,
        default=0.1,
        help=(
            "minimum Open Targets association score for inclusion in each "
            "disease's aggregate gene set; all evidence rows are retained "
            "(default: 0.1)"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing generated feature files",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Generate all initial disease feature artifacts."""

    args = parse_args(argv)
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    try:
        files = validate_inputs(input_dir)
        check_output_guard(output_dir, args.force)
        require_pyarrow()
        output_schemas = schemas()
        sources = SourceCatalog(input_dir)
        stats: Dict[str, int] = {}

        print("Parsing disease identities and ontology crosswalks...")
        catalog, product1_metadata = parse_product1(files["product1"])
        mondo_terms = parse_mondo(files["mondo"])
        hpo_labels, hpo_parents = parse_hpo(files["hpo"])
        (
            orpha_to_mondo,
            mondo_to_orpha,
            omim_to_orpha,
            efo_to_orpha,
        ) = build_crosswalks(catalog, mondo_terms)
        # Product 1 is keyed by canonical ORPHA IDs.  Keep every catalog
        # entity so literature pairs involving common diseases, broad disease
        # groups, or legacy records can still be joined to generated features.
        # Flags and disorder type/group remain in diseases.parquet, allowing
        # downstream consumers to apply narrower cohort rules when needed.
        diseases = dict(catalog)
        stats["catalog_entities"] = len(catalog)
        stats["included_diseases"] = len(diseases)
        stats["mondo_terms"] = len(mondo_terms)
        stats["hpo_terms"] = len(hpo_labels)

        phenotypes: List[Dict[str, Any]] = []
        base_genes: List[Dict[str, Any]] = []
        classifications: List[Dict[str, Any]] = []
        prevalence_rows: List[Dict[str, Any]] = []
        treatments: List[Dict[str, Any]] = []

        print("Parsing Orphadata clinical, classification, and genetic features...")
        stats["mondo_parent_rows"] = add_mondo_features(
            diseases,
            mondo_terms,
            orpha_to_mondo,
            classifications,
            sources,
            files["mondo"],
        )
        stats["preferred_parent_rows"] = parse_product7(
            files["product7"], diseases, classifications, sources
        )
        stats["classification_rows"] = parse_classifications(
            files["classifications"], diseases, classifications, sources
        )
        stats["orphadata_phenotype_rows"] = parse_product4(
            files["product4"],
            diseases,
            hpo_labels,
            hpo_parents,
            phenotypes,
            sources,
        )
        stats["orphadata_gene_rows"] = parse_product6(
            files["product6"], diseases, base_genes, sources
        )
        stats["age_records"] = parse_ages(files["ages"], diseases)
        stats["prevalence_rows"] = parse_prevalence(
            files["prevalence"], diseases, prevalence_rows, sources
        )

        print("Parsing HPO, ClinGen, and ClinVar gene evidence...")
        stats["hpo_gene_rows"] = parse_hpo_genes(
            files["hpo_genes"], diseases, omim_to_orpha, base_genes, sources
        )
        stats["clingen_gene_rows"] = parse_clingen(
            files["clingen"], diseases, mondo_to_orpha, base_genes, sources
        )
        stats["clinvar_gene_rows"] = parse_clinvar_gene_conditions(
            files["clinvar_genes"],
            diseases,
            mondo_to_orpha,
            omim_to_orpha,
            base_genes,
            sources,
        )
        base_genes = deduplicate_rows(
            base_genes,
            (
                "orpha_id",
                "gene_symbol",
                "ensembl_id",
                "ncbi_gene_id",
                "association_type",
                "classification",
                "source",
            ),
        )

        print("Parsing Open Targets disease, phenotype, target, and drug data...")
        ot_to_orpha, _, stats["opentargets_classification_rows"] = (
            parse_open_targets_diseases(
                files["ot_disease"],
                diseases,
                mondo_to_orpha,
                efo_to_orpha,
                omim_to_orpha,
                classifications,
                sources,
            )
        )
        stats["opentargets_phenotype_rows"] = parse_open_targets_phenotypes(
            files["ot_phenotype"],
            diseases,
            ot_to_orpha,
            hpo_labels,
            hpo_parents,
            phenotypes,
            sources,
        )
        targets = parse_open_targets_targets(files["ot_targets"])
        drugs = parse_open_targets_drugs(files["ot_drugs"])
        stats["opentargets_treatment_rows"] = parse_open_targets_treatments(
            files["ot_indications"],
            diseases,
            ot_to_orpha,
            drugs,
            treatments,
            sources,
        )
        stats["opentargets_mapped_disease_ids"] = len(ot_to_orpha)
        stats["opentargets_targets"] = len(targets)
        stats["opentargets_drugs"] = len(drugs)

        print("Matching regulatory designations by exact disease name...")
        name_index = build_name_index(diseases)
        ema_matched, ema_unmatched, ema_ambiguous = parse_ema_treatments(
            files["ema"], diseases, name_index, treatments, sources
        )
        fda_matched, fda_unmatched, fda_ambiguous = parse_fda_treatments(
            files["fda"], diseases, name_index, treatments, sources
        )
        stats.update(
            {
                "ema_matched": ema_matched,
                "ema_unmatched": ema_unmatched,
                "ema_ambiguous": ema_ambiguous,
                "fda_matched": fda_matched,
                "fda_unmatched": fda_unmatched,
                "fda_ambiguous": fda_ambiguous,
            }
        )

        phenotypes = deduplicate_rows(
            phenotypes,
            (
                "orpha_id",
                "hpo_id",
                "frequency",
                "qualifier",
                "sex",
                "source",
                "evidence_type",
            ),
        )
        classifications = deduplicate_rows(
            classifications,
            (
                "orpha_id",
                "source",
                "classification_id",
                "category_id",
                "source_file",
            ),
        )
        treatments = deduplicate_rows(
            treatments,
            (
                "orpha_id",
                "drug_id",
                "max_clinical_stage",
                "designation_id",
                "source",
            ),
        )
        phenotypes.sort(key=lambda row: (row["orpha_id"], row["hpo_id"], row["source"]))
        classifications.sort(
            key=lambda row: (
                row["orpha_id"],
                row["source"],
                row["classification_id"],
                row["category_id"],
            )
        )
        prevalence_rows.sort(
            key=lambda row: (
                row["orpha_id"],
                row["prevalence_type"],
                row["region"],
                row["prevalence_class"],
            )
        )
        treatments.sort(
            key=lambda row: (row["orpha_id"], row["drug_id"], row["source"])
        )

        print("Writing normalized feature tables...")
        stats["phenotypes_output_rows"] = len(phenotypes)
        stats["classifications_output_rows"] = len(classifications)
        stats["prevalence_output_rows"] = len(prevalence_rows)
        stats["treatments_output_rows"] = len(treatments)
        write_rows_atomic(
            output_dir / "phenotypes.parquet",
            phenotypes,
            output_schemas["phenotypes.parquet"],
        )
        write_rows_atomic(
            output_dir / "classifications.parquet",
            classifications,
            output_schemas["classifications.parquet"],
        )
        write_rows_atomic(
            output_dir / "prevalence.parquet",
            prevalence_rows,
            output_schemas["prevalence.parquet"],
        )
        write_rows_atomic(
            output_dir / "treatments.parquet",
            treatments,
            output_schemas["treatments.parquet"],
        )

        print("Filtering and writing Open Targets gene associations...")
        base_count, association_count = write_genes_with_open_targets(
            output_dir / "genes.parquet",
            base_genes,
            files["ot_associations"],
            diseases,
            ot_to_orpha,
            targets,
            args.min_association_score,
            output_schemas["genes.parquet"],
            sources,
        )
        stats["base_gene_output_rows"] = base_count
        stats["opentargets_gene_output_rows"] = association_count
        stats["genes_output_rows"] = base_count + association_count

        aggregate_rows = disease_rows(diseases)
        write_rows_atomic(
            output_dir / "diseases.parquet",
            aggregate_rows,
            output_schemas["diseases.parquet"],
        )
        stats["diseases_output_rows"] = len(aggregate_rows)

        used_paths: List[Path] = [
            value
            for key, value in files.items()
            if key not in {"classifications", "ot_targets", "ot_associations"}
        ]
        used_paths.extend(files["classifications"])
        used_paths.extend(files["ot_targets"])
        used_paths.extend(files["ot_associations"])
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": utc_now(),
            "input_directory": str(input_dir),
            "download_manifest_updated_at": sources.manifest.get("updated_at"),
            "parameters": {
                "min_association_score": args.min_association_score,
                "regulatory_matching": "unique exact normalized disease name or synonym",
            },
            "cohort": {
                "canonical_identifier": "ORPHA",
                "inclusion_rule": (
                    "all entities in Orphadata product 1 with a canonical ORPHA ID"
                ),
                "excluded_flags_containing": [],
                "excluded_groups": [],
                "excluded_types": [],
            },
            "source_versions": {
                "orphadata_product1": product1_metadata,
            },
            "sources": [
                {
                    "file": sources.describe(path)[0],
                    "url": sources.describe(path)[1],
                    "size": path.stat().st_size,
                }
                for path in sorted(set(used_paths))
            ],
            "outputs": {
                name: {
                    "rows": (
                        stats["diseases_output_rows"]
                        if name == "diseases.parquet"
                        else stats["phenotypes_output_rows"]
                        if name == "phenotypes.parquet"
                        else stats["genes_output_rows"]
                        if name == "genes.parquet"
                        else stats["classifications_output_rows"]
                        if name == "classifications.parquet"
                        else stats["prevalence_output_rows"]
                        if name == "prevalence.parquet"
                        else stats["treatments_output_rows"]
                    ),
                    "file": name,
                }
                for name in OUTPUT_NAMES
                if name.endswith(".parquet")
            },
            "source_counts": stats,
            "coverage": coverage(diseases),
            "warnings": [
                (
                    "Affected-sex evidence is sparse in the downloaded sources; "
                    "an empty list means unavailable, not sex-neutral."
                ),
                (
                    "No population-level epidemiology field is present in the core "
                    "download; affected_populations is therefore empty."
                ),
                (
                    "EMA and FDA records without a unique exact disease-name or "
                    "synonym match were intentionally left unmatched."
                ),
                (
                    "Open Targets gene rows below min_association_score remain in "
                    "genes.parquet but are excluded from aggregate gene_symbols."
                ),
            ],
        }
        write_json_atomic(output_dir / "feature_manifest.json", manifest)
        print(
            f"Generated {len(aggregate_rows):,} disease feature records in "
            f"{output_dir}"
        )
        return 0
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

