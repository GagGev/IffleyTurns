"""Load the static, drug-free disease records and auxiliary relation labels."""

from __future__ import annotations

import glob
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import CACHE_DIR, DATABASE_DIR, FEATURE_DIR, assign_split, normalize_curie, timed

CACHE_VERSION = 1
COHORT_GROUPS = ("Disorder", "Subtype of disorder")
EXCLUDED_FLAGS = frozenset(
    {
        "Inactive",
        "Obsolete entity",
        "Deprecated entity",
        "Historical entity",
        "Obsolete with resources",
        "Non-rare disease in Europe",
    }
)
OPEN_TARGETS_SOURCE = "opentargets_association_overall_direct"
EXCLUDED_ORPHANET_GENE_TYPES = frozenset({"Candidate gene tested in", "Biomarker tested in"})
ACCEPTED_CLINGEN = frozenset({"Definitive", "Strong", "Moderate", "Limited"})
PHENOTYPIC_ABNORMALITY = "HP:0000118"
DEFAULT_PHENOTYPE_WEIGHT = 0.5
TOKEN = re.compile(r"[A-Za-z0-9]+")


@dataclass
class Knowledge:
    hpo_labels: dict[str, str]
    hpo_parents: dict[str, tuple[str, ...]]
    hpo_alternatives: dict[str, str]
    gene_pathways: dict[str, frozenset[str]]
    pathway_labels: dict[str, str]
    ontology_parents: dict[str, frozenset[str]]
    ontology_labels: dict[str, str]
    gene_symbols: frozenset[str]
    drug_words: frozenset[str]
    _hpo_closure: dict[str, frozenset[str]] = field(default_factory=dict, repr=False)
    _ontology_closure: dict[str, frozenset[str]] = field(default_factory=dict, repr=False)

    def canonical_hpo(self, term: str) -> Optional[str]:
        term = normalize_curie(term)
        term = self.hpo_alternatives.get(term, term)
        return term if term in self.hpo_labels else None

    def hpo_ancestors(self, term: str) -> frozenset[str]:
        cached = self._hpo_closure.get(term)
        if cached is not None:
            return cached
        self._hpo_closure[term] = frozenset({term})
        result = {term}
        for parent in self.hpo_parents.get(term, ()):
            result.update(self.hpo_ancestors(parent))
        closure = frozenset(result)
        self._hpo_closure[term] = closure
        return closure

    def is_phenotypic_abnormality(self, term: str) -> bool:
        return term != PHENOTYPIC_ABNORMALITY and PHENOTYPIC_ABNORMALITY in self.hpo_ancestors(term)

    def ontology_ancestors(self, node: str) -> frozenset[str]:
        cached = self._ontology_closure.get(node)
        if cached is not None:
            return cached
        self._ontology_closure[node] = frozenset({node})
        result = {node}
        for parent in self.ontology_parents.get(node, ()):
            result.update(self.ontology_ancestors(parent))
        closure = frozenset(result)
        self._ontology_closure[node] = closure
        return closure

    def expand_ontology(self, parents: Iterable[str]) -> set[str]:
        expanded: set[str] = set()
        for parent in parents:
            expanded.update(self.ontology_ancestors(normalize_curie(parent)))
        return expanded


@dataclass
class Bundle:
    records: dict[str, dict[str, Any]]
    names: dict[str, str]
    splits: dict[str, str]
    knowledge: Knowledge
    orphanet_edges: set[tuple[str, str]]
    causal_genes: dict[str, frozenset[str]]


def _parse_obo(path: Path, prefix: str) -> dict[str, dict[str, Any]]:
    terms: dict[str, dict[str, Any]] = {}
    current: Optional[dict[str, Any]] = None
    in_term = False
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line.startswith("["):
                if current is not None and current.get("id", "").startswith(prefix):
                    terms[current["id"]] = current
                in_term = line == "[Term]"
                current = {"parents": [], "alt_ids": [], "obsolete": False, "replaced_by": None} if in_term else None
                continue
            if not in_term or current is None or not line:
                continue
            key, _, value = line.partition(": ")
            if key == "id":
                current["id"] = value.strip()
            elif key == "name":
                current["name"] = value.strip()
            elif key == "is_a":
                parent = value.split()[0]
                if parent.startswith(prefix):
                    current["parents"].append(parent)
            elif key == "alt_id":
                current["alt_ids"].append(value.strip())
            elif key == "is_obsolete" and value.strip() == "true":
                current["obsolete"] = True
            elif key == "replaced_by":
                current["replaced_by"] = value.strip()
    if current is not None and current.get("id", "").startswith(prefix):
        terms[current["id"]] = current
    return terms


def _first_parquet(folder: str) -> str:
    paths = sorted(glob.glob(str(DATABASE_DIR / "opentargets" / folder / "*.parquet")))
    if not paths:
        raise FileNotFoundError(f"No Open Targets parquet files found in {folder!r}")
    return paths[0]


def _input_fingerprint() -> list[tuple[str, int, int]]:
    paths = sorted(FEATURE_DIR.glob("*.parquet"))
    paths += [DATABASE_DIR / "hpo" / "hp.obo", DATABASE_DIR / "mondo" / "mondo.obo"]
    for folder in ("reactome", "target", "drug_molecule"):
        paths += sorted((DATABASE_DIR / "opentargets" / folder).glob("*.parquet"))
    return [(str(path.relative_to(DATABASE_DIR.parent)), path.stat().st_size, int(path.stat().st_mtime)) for path in paths]


def load_bundle(use_cache: bool = True) -> Bundle:
    cache_path = CACHE_DIR / "bundle.pkl"
    fingerprint = (CACHE_VERSION, _input_fingerprint())
    if use_cache and cache_path.is_file():
        with cache_path.open("rb") as handle:
            stored_fingerprint, bundle = pickle.load(handle)
        if stored_fingerprint == fingerprint:
            return bundle
    bundle = _build_bundle()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump((fingerprint, bundle), handle, protocol=pickle.HIGHEST_PROTOCOL)
    return bundle


def _array(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _build_bundle() -> Bundle:
    with timed("Loading the Orphanet disease catalogue"):
        diseases = pd.read_parquet(FEATURE_DIR / "diseases.parquet")
        flags = diseases["flags"].map(lambda value: set(_array(value)))
        keep = diseases["disorder_group"].isin(COHORT_GROUPS) & ~flags.map(lambda value: bool(value & EXCLUDED_FLAGS))
        cohort = diseases[keep].copy()
        cohort_ids = set(cohort["orpha_id"])
        all_names = dict(zip(diseases["orpha_id"], diseases["name"]))

    edges, direct_parents = _orphanet_tree(cohort_ids)
    knowledge = _build_knowledge(all_names)
    with timed("Loading phenotypes"):
        phenotypes = _load_phenotypes(cohort_ids, knowledge)
    with timed("Loading curated genes"):
        genes, causal_genes = _load_genes(cohort_ids)

    records: dict[str, dict[str, Any]] = {}
    names: dict[str, str] = {}
    splits: dict[str, str] = {}
    for row in cohort.itertuples(index=False):
        disease = row.orpha_id
        mondo_ids = [normalize_curie(value) for value in _array(row.mondo_ids)]
        ontology_parents = set(direct_parents.get(disease, ()))
        for mondo in mondo_ids:
            ontology_parents.update(knowledge.ontology_parents.get(mondo, ()))
        prevalence = row.prevalence_estimated_per_person
        prevalence = float(prevalence) if prevalence is not None and np.isfinite(prevalence) and prevalence > 0 else None
        description = row.description if isinstance(row.description, str) else ""
        records[disease] = {
            "name": row.name,
            "synonyms": [value for value in _array(row.synonyms) if value and value != row.name],
            "description": description,
            "phenotypes": phenotypes.get(disease, {}),
            "genes": genes.get(disease, {}),
            "inheritance": _array(row.inheritance),
            "onset": _array(row.onset),
            "prevalence": prevalence,
            "ontology_parents": sorted(ontology_parents),
        }
        names[disease] = row.name
        splits[disease] = assign_split(disease)
    return Bundle(
        records=records,
        names=names,
        splits=splits,
        knowledge=knowledge,
        orphanet_edges=edges,
        causal_genes={key: frozenset(value) for key, value in causal_genes.items()},
    )


def _orphanet_tree(cohort_ids: set[str]) -> tuple[set[tuple[str, str]], dict[str, set[str]]]:
    classifications = pd.read_parquet(
        FEATURE_DIR / "classifications.parquet", columns=["orpha_id", "source", "path_ids"]
    )
    edges: set[tuple[str, str]] = set()
    for path in classifications.loc[classifications["source"] == "orphadata_product3", "path_ids"]:
        values = _array(path)
        edges.update(zip(values[:-1], values[1:]))
    direct: dict[str, set[str]] = defaultdict(set)
    for parent, child in edges:
        if child in cohort_ids:
            direct[child].add(parent)
    return edges, direct


def _build_knowledge(all_orpha_names: dict[str, str]) -> Knowledge:
    with timed("Parsing HPO and Mondo"):
        hpo = _parse_obo(DATABASE_DIR / "hpo" / "hp.obo", "HP:")
        hpo_labels = {key: value.get("name", key) for key, value in hpo.items() if not value["obsolete"]}
        hpo_parents = {
            key: tuple(parent for parent in value["parents"] if parent in hpo_labels)
            for key, value in hpo.items()
            if not value["obsolete"]
        }
        hpo_alternatives: dict[str, str] = {}
        for term, info in hpo.items():
            hpo_alternatives.update({alt: term for alt in info["alt_ids"]})
            if info["obsolete"] and info["replaced_by"]:
                hpo_alternatives[term] = info["replaced_by"]

        mondo = _parse_obo(DATABASE_DIR / "mondo" / "mondo.obo", "MONDO:")
        ontology_parents: dict[str, frozenset[str]] = {
            key: frozenset(value["parents"]) for key, value in mondo.items() if not value["obsolete"]
        }
        ontology_labels = {
            key: value.get("name", key) for key, value in mondo.items() if not value["obsolete"]
        }

    classifications = pd.read_parquet(FEATURE_DIR / "classifications.parquet", columns=["source", "path_ids"])
    orphanet_parents: dict[str, set[str]] = defaultdict(set)
    for path in classifications.loc[classifications["source"] == "orphadata_product3", "path_ids"]:
        values = _array(path)
        for parent, child in zip(values[:-1], values[1:]):
            orphanet_parents[child].add(parent)
    for child, parents in orphanet_parents.items():
        ontology_parents[child] = frozenset(parents)
    for disease, name in all_orpha_names.items():
        ontology_labels.setdefault(disease, name)

    with timed("Loading Reactome pathways"):
        reactome = pd.read_parquet(_first_parquet("reactome"))
        pathway_ancestors = {row.id: set(_array(row.ancestors)) | {row.id} for row in reactome.itertuples()}
        pathway_labels = dict(zip(reactome["id"], reactome["label"]))
        gene_pathways: dict[str, set[str]] = defaultdict(set)
        target_files = sorted(glob.glob(str(DATABASE_DIR / "opentargets" / "target" / "*.parquet")))
        target_symbols: set[str] = set()
        for path in target_files:
            table = pq.read_table(path, columns=["approvedSymbol", "pathways"]).to_pandas()
            for row in table.itertuples(index=False):
                if not row.approvedSymbol:
                    continue
                target_symbols.add(row.approvedSymbol)
                for entry in _array(row.pathways):
                    pathway = entry.get("pathwayId")
                    if pathway:
                        gene_pathways[row.approvedSymbol].update(pathway_ancestors.get(pathway, {pathway}))

    gene_symbols = pq.read_table(
        FEATURE_DIR / "genes.parquet",
        columns=["gene_symbol"],
        filters=[("source", "!=", OPEN_TARGETS_SOURCE)],
    ).column(0).to_pylist()
    molecules = pd.read_parquet(_first_parquet("drug_molecule"), columns=["name"])
    biomedical_words = {
        token.lower()
        for label in [*hpo_labels.values(), *ontology_labels.values()]
        for token in TOKEN.findall(str(label))
    }
    drug_words = {
        tokens[0]
        for value in molecules["name"].dropna()
        if len(tokens := TOKEN.findall(str(value).lower())) == 1 and len(tokens[0]) >= 5 and tokens[0] not in biomedical_words
    }
    return Knowledge(
        hpo_labels=hpo_labels,
        hpo_parents=hpo_parents,
        hpo_alternatives=hpo_alternatives,
        gene_pathways={key: frozenset(value) for key, value in gene_pathways.items()},
        pathway_labels=pathway_labels,
        ontology_parents=ontology_parents,
        ontology_labels=ontology_labels,
        gene_symbols=frozenset(value for value in set(gene_symbols) | target_symbols if value),
        drug_words=frozenset(drug_words),
    )


def _load_phenotypes(cohort_ids: set[str], knowledge: Knowledge) -> dict[str, dict[str, float]]:
    table = pd.read_parquet(
        FEATURE_DIR / "phenotypes.parquet", columns=["orpha_id", "hpo_id", "frequency_weight", "present"]
    )
    table = table[table["present"] & table["orpha_id"].isin(cohort_ids)]
    weights = table["frequency_weight"].fillna(DEFAULT_PHENOTYPE_WEIGHT)
    result: dict[str, dict[str, float]] = defaultdict(dict)
    for disease, term, weight in zip(table["orpha_id"], table["hpo_id"], weights):
        canonical = knowledge.canonical_hpo(term)
        if weight <= 0 or canonical is None or not knowledge.is_phenotypic_abnormality(canonical):
            continue
        result[disease][canonical] = max(float(weight), result[disease].get(canonical, 0.0))
    return dict(result)


def _load_genes(cohort_ids: set[str]) -> tuple[dict[str, dict[str, float]], dict[str, set[str]]]:
    table = pq.read_table(
        FEATURE_DIR / "genes.parquet",
        columns=["orpha_id", "gene_symbol", "association_type", "classification", "source"],
        filters=[("source", "!=", OPEN_TARGETS_SOURCE)],
    ).to_pandas()
    table = table[table["orpha_id"].isin(cohort_ids) & table["gene_symbol"].fillna("").ne("")]
    keep = (
        ((table["source"] == "orphadata_product6") & ~table["association_type"].isin(EXCLUDED_ORPHANET_GENE_TYPES))
        | ((table["source"] == "clingen_gene_validity") & table["classification"].isin(ACCEPTED_CLINGEN))
        | table["source"].isin(["hpo_genes_to_disease", "clinvar_gene_condition"])
    )
    genes: dict[str, dict[str, float]] = defaultdict(dict)
    for disease, symbol in zip(table.loc[keep, "orpha_id"], table.loc[keep, "gene_symbol"]):
        genes[disease][symbol] = 1.0
    causal_mask = (
        (table["source"] == "orphadata_product6")
        & table["association_type"].fillna("").str.startswith("Disease-causing germline")
        & (table["classification"] == "Assessed")
    )
    causal: dict[str, set[str]] = defaultdict(set)
    for disease, symbol in zip(table.loc[causal_mask, "orpha_id"], table.loc[causal_mask, "gene_symbol"]):
        causal[disease].add(symbol)
    return dict(genes), dict(causal)
