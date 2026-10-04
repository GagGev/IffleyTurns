"""Load disease records, reference knowledge, and curated relation sources.

Only downloaded database content is used: the normalized tables in
``.data/features`` (built by ``generate_features.py`` from Orphadata, HPO,
Mondo, ClinGen, ClinVar, Open Targets, FDA, and EMA) plus reference files in
``.data/databases``.  The literature review is never read.

Three kinds of objects are produced:

* ``records`` -- one plain dictionary per cohort disease.  A new disease is
  described with exactly the same keys, so every encoder is inductive.
* ``Knowledge`` -- reference resources that turn raw annotations into
  features: HPO and disease-ontology hierarchies, Reactome pathways of genes,
  and molecular targets of drugs.
* ``LabelSources`` -- curated relations used only to build evaluation
  benchmarks (Orphanet tree edges, Orphanet causal genes, and drugs tested in
  phase-2-or-later trials).
"""

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

CACHE_VERSION = 4
COHORT_GROUPS = ("Disorder", "Subtype of disorder")
EXCLUDED_FLAGS = frozenset(
    {
        "Inactive",
        "Obsolete entity",
        "Deprecated entity",
        "Historical entity",
        "Obsolete with resources",
        # PROJECT.md defines rare as < 1/2,000; Orphanet flags exceptions.
        "Non-rare disease in Europe",
    }
)
PHENOTYPIC_ABNORMALITY = "HP:0000118"
OPEN_TARGETS_ASSOCIATION_SOURCE = "opentargets_association_overall_direct"
MIN_OPEN_TARGETS_SCORE = 0.5
EXCLUDED_ORPHANET_GENE_TYPES = frozenset({"Candidate gene tested in", "Biomarker tested in"})
ACCEPTED_CLINGEN = frozenset({"Definitive", "Strong", "Moderate", "Limited"})
TRIAL_STAGE_WEIGHTS = {
    "APPROVAL": 1.0,
    "PREAPPROVAL": 0.9,
    "PHASE_3": 0.8,
    "PHASE_2_3": 0.8,
    "PHASE_2": 0.6,
    "PHASE_1_2": 0.5,
    "PHASE_1": 0.4,
    "EARLY_PHASE_1": 0.4,
    "IND": 0.3,
    "PRECLINICAL": 0.3,
    "UNKNOWN": 0.3,
}
LABEL_TRIAL_STAGES = frozenset({"APPROVAL", "PREAPPROVAL", "PHASE_3", "PHASE_2_3", "PHASE_2"})
REGULATORY_DESIGNATION_WEIGHT = 0.6
INHERITANCE_ABBREVIATIONS = {
    "ar": "autosomal recessive",
    "ad": "autosomal dominant",
    "xlr": "x-linked recessive",
    "xld": "x-linked dominant",
    "xl": "x-linked",
}
DEFAULT_PHENOTYPE_WEIGHT = 0.5

RECORD_KEYS = (
    "name",
    "synonyms",
    "description",
    "phenotypes",
    "genes",
    "ot_genes",
    "inheritance",
    "onset",
    "prevalence",
    "drugs",
    "ontology_parents",
)


@dataclass
class Knowledge:
    """Reference resources needed to encode any disease, including new ones."""

    hpo_labels: dict[str, str]
    hpo_parents: dict[str, tuple[str, ...]]
    hpo_alternatives: dict[str, str]
    gene_pathways: dict[str, frozenset[str]]
    pathway_labels: dict[str, str]
    drug_targets: dict[str, frozenset[str]]
    drug_labels: dict[str, str]
    drug_parent: dict[str, str]
    drug_name_index: dict[str, str]
    ontology_parents: dict[str, frozenset[str]]
    ontology_labels: dict[str, str]
    orphanet_heads: frozenset[str]
    gene_symbols: frozenset[str]
    _hpo_closure: dict[str, frozenset[str]] = field(default_factory=dict, repr=False)
    _ontology_closure: dict[str, frozenset[str]] = field(default_factory=dict, repr=False)

    def canonical_hpo(self, term: str) -> Optional[str]:
        term = normalize_curie(term)
        term = self.hpo_alternatives.get(term, term)
        return term if term in self.hpo_labels else None

    def hpo_ancestors(self, term: str) -> frozenset[str]:
        """Return the term and its ancestors below 'Phenotypic abnormality'."""

        cached = self._hpo_closure.get(term)
        if cached is not None:
            return cached
        result = {term}
        for parent in self.hpo_parents.get(term, ()):
            result |= self.hpo_ancestors(parent)
        closure = frozenset(result)
        self._hpo_closure[term] = closure
        return closure

    def is_phenotypic_abnormality(self, term: str) -> bool:
        return PHENOTYPIC_ABNORMALITY in self.hpo_ancestors(term) and term != PHENOTYPIC_ABNORMALITY

    def ontology_ancestors(self, node: str) -> frozenset[str]:
        """Return a disease-ontology node and all its Orphanet/Mondo ancestors."""

        cached = self._ontology_closure.get(node)
        if cached is not None:
            return cached
        self._ontology_closure[node] = frozenset({node})  # Guards against cycles.
        result = {node}
        for parent in self.ontology_parents.get(node, ()):
            result |= self.ontology_ancestors(parent)
        closure = frozenset(result)
        self._ontology_closure[node] = closure
        return closure

    def expand_ontology(self, parents: Iterable[str]) -> set[str]:
        expanded: set[str] = set()
        for parent in parents:
            expanded |= self.ontology_ancestors(normalize_curie(parent))
        return expanded

    def resolve_drug(self, value: str) -> Optional[str]:
        """Map a ChEMBL ID or a drug name/synonym/trade name to a parent ChEMBL ID."""

        value = str(value).strip()
        if value.upper().startswith("CHEMBL"):
            chembl = value.upper()
            return self.drug_parent.get(chembl, chembl)
        return self.drug_name_index.get(normalize_drug_name(value))


@dataclass
class LabelSources:
    """Curated relations used to build evaluation benchmarks, never as inputs."""

    orphanet_edges: set[tuple[str, str]]
    causal_genes: dict[str, frozenset[str]]
    # disease -> drug -> Open Targets clinical-indication record IDs.  One
    # Open Targets disease can cross-reference several Orphanet entities; the
    # record IDs let benchmarks ignore drug "sharing" created by that mapping.
    trial_drugs: dict[str, dict[str, frozenset[str]]]


@dataclass
class Bundle:
    records: dict[str, dict[str, Any]]
    meta: pd.DataFrame
    knowledge: Knowledge
    labels: LabelSources
    v1_sets: dict[str, dict[str, Any]]


def normalize_drug_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _parse_obo(path: Path, prefix: str) -> dict[str, dict[str, Any]]:
    """Parse the subset of OBO needed here: names, is_a, alt IDs, obsolescence."""

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


def _input_fingerprint() -> list[tuple[str, int, int]]:
    paths = sorted(FEATURE_DIR.glob("*.parquet")) + [
        DATABASE_DIR / "hpo" / "hp.obo",
        DATABASE_DIR / "mondo" / "mondo.obo",
    ]
    paths += sorted(Path(p) for p in glob.glob(str(DATABASE_DIR / "opentargets" / "target" / "*.parquet")))
    for name in ("reactome", "drug_mechanism_of_action", "drug_molecule"):
        paths += sorted(Path(p) for p in glob.glob(str(DATABASE_DIR / "opentargets" / name / "*.parquet")))
    return [(str(p.relative_to(DATABASE_DIR.parent)), p.stat().st_size, int(p.stat().st_mtime)) for p in paths]


def load_bundle(use_cache: bool = True) -> Bundle:
    """Load (and cache) everything needed by the v2 models."""

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


def _build_bundle() -> Bundle:
    with timed("Loading Orphanet disease catalog"):
        diseases = pd.read_parquet(FEATURE_DIR / "diseases.parquet")
        flags = diseases["flags"].map(lambda values: set(values.tolist()) if values is not None else set())
        in_cohort = diseases["disorder_group"].isin(COHORT_GROUPS) & ~flags.map(
            lambda values: bool(values & EXCLUDED_FLAGS)
        )
        cohort = diseases[in_cohort].copy()
        cohort_ids = set(cohort["orpha_id"])
        all_names = dict(zip(diseases["orpha_id"], diseases["name"]))

    knowledge = _build_knowledge(all_names)
    labels_edges, direct_parents = _orphanet_tree(cohort_ids)

    with timed("Loading phenotype annotations"):
        phenotypes = _load_phenotypes(cohort_ids, knowledge)
    with timed("Loading gene annotations"):
        curated_genes, causal_genes, ot_genes = _load_genes(cohort_ids)
    with timed("Loading treatment annotations"):
        drugs, trial_drugs = _load_drugs(cohort_ids, knowledge)

    mondo_parents = knowledge.ontology_parents
    records: dict[str, dict[str, Any]] = {}
    v1_sets: dict[str, dict[str, Any]] = {}
    meta_rows = []
    for row in cohort.itertuples(index=False):
        orpha = row.orpha_id
        mondo_ids = [normalize_curie(x) for x in (row.mondo_ids if row.mondo_ids is not None else [])]
        ontology_parents = set(direct_parents.get(orpha, ()))
        for mondo in mondo_ids:
            ontology_parents |= set(mondo_parents.get(mondo, ()))
        prevalence = row.prevalence_estimated_per_person
        prevalence = float(prevalence) if prevalence is not None and np.isfinite(prevalence) and prevalence > 0 else None
        records[orpha] = {
            "name": row.name,
            "synonyms": [s for s in (row.synonyms if row.synonyms is not None else []) if s and s != row.name],
            "description": row.description or "",
            "phenotypes": phenotypes.get(orpha, {}),
            "genes": curated_genes.get(orpha, {}),
            "ot_genes": ot_genes.get(orpha, {}),
            "inheritance": list(row.inheritance if row.inheritance is not None else []),
            "onset": list(row.onset if row.onset is not None else []),
            "prevalence": prevalence,
            "drugs": drugs.get(orpha, {}),
            "ontology_parents": sorted(ontology_parents),
        }
        v1_sets[orpha] = {
            column: frozenset(str(x) for x in (getattr(row, column) if getattr(row, column) is not None else []))
            for column in (
                "hpo_ids",
                "gene_symbols",
                "category_ids",
                "body_system_ids",
                "inheritance",
                "onset",
                "approved_drug_ids",
            )
        }
        v1_sets[orpha]["prevalence"] = prevalence
        top = list(row.preferential_parent_names) if row.preferential_parent_names is not None else []
        meta_rows.append(
            {
                "orpha_id": orpha,
                "name": row.name,
                "disorder_type": row.disorder_type,
                "disorder_group": row.disorder_group,
                "top_category": top[0] if top else "Unclassified",
                "expert_url": row.expert_url,
                "mondo_ids": ";".join(mondo_ids),
                "split": assign_split(orpha),
            }
        )

    meta = pd.DataFrame(meta_rows).set_index("orpha_id").sort_index()
    labels = LabelSources(
        orphanet_edges=labels_edges,
        causal_genes={k: frozenset(v) for k, v in causal_genes.items()},
        trial_drugs={k: {drug: frozenset(ids) for drug, ids in v.items()} for k, v in trial_drugs.items()},
    )
    return Bundle(records=records, meta=meta, knowledge=knowledge, labels=labels, v1_sets=v1_sets)


def _orphanet_tree(cohort_ids: set[str]) -> tuple[set[tuple[str, str]], dict[str, set[str]]]:
    """Return Orphanet (product 3) parent->child edges and cohort direct parents."""

    classifications = pd.read_parquet(
        FEATURE_DIR / "classifications.parquet", columns=["orpha_id", "source", "path_ids"]
    )
    product3 = classifications[classifications["source"] == "orphadata_product3"]
    edges: set[tuple[str, str]] = set()
    for path in product3["path_ids"]:
        path = list(path)
        edges.update(zip(path[:-1], path[1:]))
    direct_parents: dict[str, set[str]] = defaultdict(set)
    for parent, child in edges:
        if child in cohort_ids:
            direct_parents[child].add(parent)
    return edges, direct_parents


def _build_knowledge(all_orpha_names: dict[str, str]) -> Knowledge:
    with timed("Parsing HPO ontology"):
        hpo = _parse_obo(DATABASE_DIR / "hpo" / "hp.obo", "HP:")
        hpo_labels = {k: v.get("name", k) for k, v in hpo.items() if not v["obsolete"]}
        hpo_parents = {k: tuple(p for p in v["parents"] if p in hpo_labels) for k, v in hpo.items() if not v["obsolete"]}
        alternatives = {}
        for term, info in hpo.items():
            for alt in info["alt_ids"]:
                alternatives[alt] = term
            if info["obsolete"] and info["replaced_by"]:
                alternatives[term] = info["replaced_by"]

    with timed("Parsing Mondo ontology"):
        mondo = _parse_obo(DATABASE_DIR / "mondo" / "mondo.obo", "MONDO:")
        ontology_parents: dict[str, frozenset[str]] = {
            k: frozenset(v["parents"]) for k, v in mondo.items() if not v["obsolete"]
        }
        ontology_labels = {k: v.get("name", k) for k, v in mondo.items() if not v["obsolete"]}

    classifications = pd.read_parquet(FEATURE_DIR / "classifications.parquet", columns=["source", "path_ids"])
    orphanet_parents: dict[str, set[str]] = defaultdict(set)
    heads: set[str] = set()
    for path in classifications.loc[classifications["source"] == "orphadata_product3", "path_ids"]:
        path = list(path)
        heads.add(path[0])
        for parent, child in zip(path[:-1], path[1:]):
            orphanet_parents[child].add(parent)
    for child, parents in orphanet_parents.items():
        ontology_parents[child] = frozenset(parents)
    for orpha, name in all_orpha_names.items():
        ontology_labels.setdefault(orpha, name)

    with timed("Loading Open Targets targets, pathways, and drugs"):
        reactome = pd.read_parquet(glob.glob(str(DATABASE_DIR / "opentargets" / "reactome" / "*.parquet"))[0])
        pathway_ancestors = {row.id: set(row.ancestors) | {row.id} for row in reactome.itertuples()}
        pathway_labels = dict(zip(reactome["id"], reactome["label"]))
        gene_pathways: dict[str, set[str]] = defaultdict(set)
        ensembl_to_symbol: dict[str, str] = {}
        for path in sorted(glob.glob(str(DATABASE_DIR / "opentargets" / "target" / "*.parquet"))):
            table = pq.read_table(path, columns=["id", "approvedSymbol", "pathways"]).to_pandas()
            for row in table.itertuples(index=False):
                if not row.approvedSymbol:
                    continue
                ensembl_to_symbol[row.id] = row.approvedSymbol
                for entry in row.pathways if row.pathways is not None else []:
                    pathway = entry.get("pathwayId")
                    if pathway:
                        gene_pathways[row.approvedSymbol] |= pathway_ancestors.get(pathway, {pathway})

        molecules = pd.read_parquet(
            glob.glob(str(DATABASE_DIR / "opentargets" / "drug_molecule" / "*.parquet"))[0],
            columns=["id", "name", "parentId", "synonyms", "tradeNames"],
        )
        drug_parent = {
            row.id: row.parentId for row in molecules.itertuples(index=False) if row.parentId and row.parentId != row.id
        }
        drug_labels = {row.id: (row.name or row.id) for row in molecules.itertuples(index=False)}
        name_candidates: dict[str, set[str]] = defaultdict(set)
        for row in molecules.itertuples(index=False):
            parent = drug_parent.get(row.id, row.id)
            names = [row.name]
            for column in (row.synonyms, row.tradeNames):
                names += [entry.get("label") for entry in (column if column is not None else [])]
            for name in names:
                if name:
                    name_candidates[normalize_drug_name(name)].add(parent)
        drug_name_index = {k: next(iter(v)) for k, v in name_candidates.items() if k and len(v) == 1}

        mechanisms = pd.read_parquet(
            glob.glob(str(DATABASE_DIR / "opentargets" / "drug_mechanism_of_action" / "*.parquet"))[0],
            columns=["chemblIds", "targets"],
        )
        drug_targets: dict[str, set[str]] = defaultdict(set)
        for row in mechanisms.itertuples(index=False):
            symbols = {ensembl_to_symbol[t] for t in (row.targets if row.targets is not None else []) if t in ensembl_to_symbol}
            for chembl in row.chemblIds if row.chemblIds is not None else []:
                drug_targets[drug_parent.get(chembl, chembl)] |= symbols

    gene_symbols = pq.read_table(
        FEATURE_DIR / "genes.parquet",
        columns=["gene_symbol"],
        filters=[("source", "!=", OPEN_TARGETS_ASSOCIATION_SOURCE)],
    ).column(0).to_pylist()
    return Knowledge(
        hpo_labels=hpo_labels,
        hpo_parents=hpo_parents,
        hpo_alternatives=alternatives,
        gene_pathways={k: frozenset(v) for k, v in gene_pathways.items()},
        pathway_labels=pathway_labels,
        drug_targets={k: frozenset(v) for k, v in drug_targets.items() if v},
        drug_labels=drug_labels,
        drug_parent=drug_parent,
        drug_name_index=drug_name_index,
        ontology_parents=ontology_parents,
        ontology_labels=ontology_labels,
        orphanet_heads=frozenset(heads),
        gene_symbols=frozenset(s for s in set(gene_symbols) | set(ensembl_to_symbol.values()) if s),
    )


def _load_phenotypes(cohort_ids: set[str], knowledge: Knowledge) -> dict[str, dict[str, float]]:
    table = pd.read_parquet(
        FEATURE_DIR / "phenotypes.parquet", columns=["orpha_id", "hpo_id", "frequency_weight", "present"]
    )
    table = table[table["present"] & table["orpha_id"].isin(cohort_ids)]
    weights = table["frequency_weight"].fillna(DEFAULT_PHENOTYPE_WEIGHT)
    result: dict[str, dict[str, float]] = defaultdict(dict)
    for orpha, term, weight in zip(table["orpha_id"], table["hpo_id"], weights):
        if weight <= 0:
            continue
        term = knowledge.canonical_hpo(term)
        if term is None or not knowledge.is_phenotypic_abnormality(term):
            continue
        current = result[orpha].get(term, 0.0)
        if weight > current:
            result[orpha][term] = float(weight)
    return dict(result)


def _load_genes(
    cohort_ids: set[str],
) -> tuple[dict[str, dict[str, float]], dict[str, set[str]], dict[str, dict[str, float]]]:
    curated = pq.read_table(
        FEATURE_DIR / "genes.parquet",
        columns=["orpha_id", "gene_symbol", "association_type", "classification", "source"],
        filters=[("source", "!=", OPEN_TARGETS_ASSOCIATION_SOURCE)],
    ).to_pandas()
    curated = curated[curated["orpha_id"].isin(cohort_ids) & curated["gene_symbol"].fillna("").ne("")]
    keep = (
        ((curated["source"] == "orphadata_product6") & ~curated["association_type"].isin(EXCLUDED_ORPHANET_GENE_TYPES))
        | ((curated["source"] == "clingen_gene_validity") & curated["classification"].isin(ACCEPTED_CLINGEN))
        | curated["source"].isin(["hpo_genes_to_disease", "clinvar_gene_condition"])
    )
    genes: dict[str, dict[str, float]] = defaultdict(dict)
    for orpha, symbol in zip(curated.loc[keep, "orpha_id"], curated.loc[keep, "gene_symbol"]):
        genes[orpha][symbol] = 1.0

    causal_mask = (
        (curated["source"] == "orphadata_product6")
        & curated["association_type"].fillna("").str.startswith("Disease-causing germline")
        & (curated["classification"] == "Assessed")
    )
    causal: dict[str, set[str]] = defaultdict(set)
    for orpha, symbol in zip(curated.loc[causal_mask, "orpha_id"], curated.loc[causal_mask, "gene_symbol"]):
        causal[orpha].add(symbol)

    associations = pq.read_table(
        FEATURE_DIR / "genes.parquet",
        columns=["orpha_id", "gene_symbol", "association_score"],
        filters=[
            ("source", "=", OPEN_TARGETS_ASSOCIATION_SOURCE),
            ("association_score", ">=", MIN_OPEN_TARGETS_SCORE),
        ],
    ).to_pandas()
    associations = associations[associations["orpha_id"].isin(cohort_ids) & associations["gene_symbol"].fillna("").ne("")]
    ot_genes: dict[str, dict[str, float]] = defaultdict(dict)
    for orpha, symbol, score in associations.itertuples(index=False):
        if score > ot_genes[orpha].get(symbol, 0.0):
            ot_genes[orpha][symbol] = float(score)
    return dict(genes), dict(causal), dict(ot_genes)


def _load_drugs(
    cohort_ids: set[str], knowledge: Knowledge
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, set[str]]]]:
    treatments = pd.read_parquet(
        FEATURE_DIR / "treatments.parquet",
        columns=["orpha_id", "drug_id", "drug_name", "max_clinical_stage", "designation_id", "source"],
    )
    treatments = treatments[treatments["orpha_id"].isin(cohort_ids)]
    drugs: dict[str, dict[str, float]] = defaultdict(dict)
    trial_drugs: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in treatments.itertuples(index=False):
        if row.source == "opentargets_clinical_indication":
            chembl = knowledge.drug_parent.get(row.drug_id, row.drug_id)
            weight = TRIAL_STAGE_WEIGHTS.get(row.max_clinical_stage or "UNKNOWN", 0.3)
            if row.max_clinical_stage in LABEL_TRIAL_STAGES:
                trial_drugs[row.orpha_id][chembl].add(row.designation_id)
        else:
            chembl = knowledge.resolve_drug(row.drug_name or "")
            weight = REGULATORY_DESIGNATION_WEIGHT
        if chembl and weight > drugs[row.orpha_id].get(chembl, 0.0):
            drugs[row.orpha_id][chembl] = weight
    return dict(drugs), dict(trial_drugs)


def empty_record(name: str = "") -> dict[str, Any]:
    return {
        "name": name,
        "synonyms": [],
        "description": "",
        "phenotypes": {},
        "genes": {},
        "ot_genes": {},
        "inheritance": [],
        "onset": [],
        "prevalence": None,
        "drugs": {},
        "ontology_parents": [],
    }


def record_from_user_input(document: dict[str, Any], knowledge: Knowledge) -> tuple[dict[str, Any], list[str]]:
    """Convert a user JSON description of a (new) disease into a record.

    Lists may be given for phenotypes, genes, and drugs; dictionaries map items
    to weights.  Unknown identifiers are reported instead of silently kept.
    """

    record = empty_record(str(document.get("name", "")).strip())
    warnings: list[str] = []
    record["synonyms"] = [str(s) for s in document.get("synonyms", [])]
    record["description"] = str(document.get("description", ""))

    def weighted(values: Any, default: float = 1.0) -> dict[str, float]:
        if isinstance(values, dict):
            return {str(k): float(v) for k, v in values.items()}
        return {str(v): default for v in values or []}

    for term, weight in weighted(document.get("phenotypes", document.get("hpo_ids", [])), 0.5).items():
        canonical = knowledge.canonical_hpo(term)
        if canonical is None or not knowledge.is_phenotypic_abnormality(canonical):
            warnings.append(f"Ignored phenotype {term!r}: not an HPO phenotypic-abnormality term")
            continue
        record["phenotypes"][canonical] = max(weight, record["phenotypes"].get(canonical, 0.0))
    for symbol, weight in weighted(document.get("genes", [])).items():
        if symbol not in knowledge.gene_symbols:
            warnings.append(f"Gene {symbol!r} is not a known HGNC/Open Targets symbol; kept as given")
        record["genes"][symbol] = weight
    record["ot_genes"] = weighted(document.get("ot_genes", {}))
    for drug, weight in weighted(document.get("drugs", []), REGULATORY_DESIGNATION_WEIGHT).items():
        chembl = knowledge.resolve_drug(drug)
        if chembl is None:
            warnings.append(f"Ignored drug {drug!r}: no unique ChEMBL match")
            continue
        record["drugs"][chembl] = max(weight, record["drugs"].get(chembl, 0.0))
    from modalities import INHERITANCE_MAP, ONSET_BINS

    inheritance_names = {**{key: key for key in INHERITANCE_MAP}, **INHERITANCE_ABBREVIATIONS}
    for value in document.get("inheritance", []):
        name = inheritance_names.get(str(value).strip().lower())
        if name is None:
            warnings.append(f"Ignored inheritance {value!r}: use one of {sorted(INHERITANCE_MAP)} or AR/AD/XLR/XLD")
        else:
            record["inheritance"].append(name)
    onset_names = {bin_.lower(): bin_ for bin_ in (*ONSET_BINS, "All ages")}
    for value in document.get("onset", []):
        name = onset_names.get(str(value).strip().lower())
        if name is None:
            warnings.append(f"Ignored onset {value!r}: use one of {list(onset_names.values())}")
        else:
            record["onset"].append(name)
    prevalence = document.get("prevalence")
    if prevalence not in (None, ""):
        try:
            value = float(prevalence)
        except (TypeError, ValueError):
            value = 0.0
        if value > 0:
            record["prevalence"] = value
        else:
            warnings.append(f"Ignored prevalence {prevalence!r}: give a positive fraction, e.g. 1e-5 for 1 in 100,000")
    parents = [normalize_curie(p) for p in document.get("ontology_parents", document.get("parents", []))]
    for parent in parents:
        if parent not in knowledge.ontology_labels:
            warnings.append(f"Ontology parent {parent!r} is unknown and will not match other diseases")
    record["ontology_parents"] = parents
    return record, warnings
