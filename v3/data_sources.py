"""Load disease records, reference knowledge, and curated relation sources.

Only downloaded database content is used: the normalized tables in
``.data/features`` (built by ``generate_features.py`` from Orphadata, HPO,
Mondo, ClinGen, and ClinVar) plus reference files in ``.data/databases``.
The literature review is never read.

Unlike v2, no undated drug information enters the records.  Open Targets
clinical indications and Open Targets association scores (which aggregate
ChEMBL drug evidence) cannot be restricted to what was known at a past date,
so they would leak future drug development into a prospective evaluation.
Drug inputs come only from dated FDA/EMA orphan designations and are attached
per cutoff date by ``regulatory.with_history``.

Three kinds of objects are produced:

* ``records`` -- one plain dictionary per cohort disease.  A new disease is
  described with exactly the same keys, so every encoder is inductive.
* ``Knowledge`` -- reference resources that turn raw annotations into
  features: HPO and disease-ontology hierarchies, Reactome pathways of genes,
  and molecular targets of drugs.
* ``LabelSources`` -- curated relations used only as auxiliary training
  signals (Orphanet tree edges and Orphanet causal genes).
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

CACHE_VERSION = 2
COHORT_GROUPS = ("Disorder", "Subtype of disorder")
MAX_GROUP_MEMBERS = 200
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
EXCLUDED_ORPHANET_GENE_TYPES = frozenset({"Candidate gene tested in", "Biomarker tested in"})
ACCEPTED_CLINGEN = frozenset({"Definitive", "Strong", "Moderate", "Limited"})
DESIGNATED_DRUG_WEIGHT = 1.0
SALT_WORDS = frozenset(
    """acetate hydrochloride hcl sodium potassium calcium magnesium mesylate mesilate maleate citrate phosphate
    sulfate sulphate tartrate bitartrate fumarate succinate bromide chloride besylate besilate tosylate
    dihydrochloride monohydrate dihydrate trihydrate hydrate hemihydrate sesquihydrate anhydrous disodium
    monosodium dipotassium meglumine lactate hydrobromide nitrate oxalate malate gluconate stearate palmitate
    dipropionate propionate valerate pivalate embonate pamoate trifluoroacetate tromethamine edisylate
    napadisylate xinafoate hyclate cypionate enanthate decanoate salt base free""".split()
)
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
        """Map a ChEMBL ID or a drug name/synonym/trade name to a parent ChEMBL ID.

        Salt and hydrate words ("octreotide acetate") and parenthesized forms
        ("Octreotide (acetate)") are tried as well, since regulators name the
        marketed salt while ChEMBL parents carry the active moiety.
        """

        value = str(value).strip()
        if value.upper().startswith("CHEMBL"):
            chembl = value.upper()
            return self.drug_parent.get(chembl, chembl)
        for candidate in drug_name_variants(value):
            chembl = self.drug_name_index.get(normalize_drug_name(candidate))
            if chembl:
                return chembl
        return None

    def drug_identifier(self, value: str) -> Optional[str]:
        """Resolved ChEMBL parent ID, or a ``NAME:`` key for unresolved names."""

        chembl = self.resolve_drug(value)
        if chembl:
            return chembl
        variants = drug_name_variants(value)
        key = normalize_drug_name(variants[-1] if variants else value)
        return f"NAME:{key}" if len(key) >= 3 else None


@dataclass
class LabelSources:
    """Curated relations used only as auxiliary training signals, never as inputs."""

    orphanet_edges: set[tuple[str, str]]
    causal_genes: dict[str, frozenset[str]]


@dataclass
class Bundle:
    records: dict[str, dict[str, Any]]
    meta: pd.DataFrame
    knowledge: Knowledge
    labels: LabelSources
    v1_sets: dict[str, dict[str, Any]]
    # Orphanet groups that may become nodes (see ``attach_groups``).
    group_records: dict[str, dict[str, Any]] = field(default_factory=dict)
    group_v1: dict[str, dict[str, Any]] = field(default_factory=dict)
    group_members: dict[str, frozenset[str]] = field(default_factory=dict)
    group_meta: pd.DataFrame = field(default_factory=pd.DataFrame)

    def is_group(self, disease: str) -> bool:
        return disease in self.group_members


def normalize_drug_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def drug_name_variants(value: str) -> list[str]:
    """Name variants from most to least literal: as given, without the
    parenthesized part, the parenthesized part alone, and without salt words."""

    value = str(value).strip()
    variants = [value]
    outside = re.sub(r"\([^)]*\)", " ", value).strip()
    inside = re.findall(r"\(([^)]*)\)", value)
    variants.append(outside)
    variants += [part for part in inside if len(part) > 3]
    words = re.findall(r"[A-Za-z0-9\-]+", outside.lower())
    kept = [w for w in words if w not in SALT_WORDS]
    if kept and len(kept) < len(words):
        variants.append(" ".join(kept))
    seen: list[str] = []
    for variant in variants:
        if variant and variant not in seen:
            seen.append(variant)
    return seen


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
    """Load (and cache) everything needed by the v3 models."""

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
        curated_genes, causal_genes = _load_genes(cohort_ids)

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
            "inheritance": list(row.inheritance if row.inheritance is not None else []),
            "onset": list(row.onset if row.onset is not None else []),
            "prevalence": prevalence,
            "drugs": {},
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
    meta["level"] = "disease"
    labels = LabelSources(
        orphanet_edges=labels_edges,
        causal_genes={k: frozenset(v) for k, v in causal_genes.items()},
    )
    bundle = Bundle(records=records, meta=meta, knowledge=knowledge, labels=labels, v1_sets=v1_sets)
    with timed("Aggregating Orphanet group records"):
        groups = diseases[diseases["disorder_group"].eq("Group of disorders") & ~flags.map(lambda v: bool(v & EXCLUDED_FLAGS))]
        _add_group_records(bundle, groups, labels_edges)
    return bundle


def _mean_weights(maps: list[dict[str, float]]) -> dict[str, float]:
    total: dict[str, float] = defaultdict(float)
    for mapping in maps:
        for key, value in mapping.items():
            total[key] += value
    return {key: value / len(maps) for key, value in total.items()}


def _add_group_records(bundle: Bundle, groups: pd.DataFrame, edges: set[tuple[str, str]]) -> None:
    """Records for Orphanet groups, built from their member diseases.

    Regulators often designate a drug for a whole disease family ("spinal
    muscular atrophy", "acute myeloid leukemia").  Such a group becomes a graph
    node only if it receives a designation (``attach_groups``).  Its
    phenotypes and genes are the member diseases' annotations averaged over
    members, so features shared by most members dominate; inheritance and onset
    are the union over members; prevalence is the group's own estimate.
    """

    children: dict[str, set[str]] = defaultdict(set)
    parents: dict[str, set[str]] = defaultdict(set)
    for parent, child in edges:
        children[parent].add(child)
        parents[child].add(parent)
    heads = bundle.knowledge.orphanet_heads
    mondo_parents = bundle.knowledge.ontology_parents
    cohort = set(bundle.records)
    rows = []
    for row in groups.itertuples(index=False):
        orpha = row.orpha_id
        if orpha in heads:
            continue
        members: set[str] = set()
        stack, seen = [orpha], {orpha}
        while stack:
            for child in children.get(stack.pop(), ()):
                if child not in seen:
                    seen.add(child)
                    stack.append(child)
                    if child in cohort:
                        members.add(child)
        if not 2 <= len(members) <= MAX_GROUP_MEMBERS:
            continue
        member_records = [bundle.records[m] for m in sorted(members)]
        mondo_ids = [normalize_curie(x) for x in (row.mondo_ids if row.mondo_ids is not None else [])]
        ontology_parents = set(parents.get(orpha, ()))
        for mondo in mondo_ids:
            ontology_parents |= set(mondo_parents.get(mondo, ()))
        prevalence = row.prevalence_estimated_per_person
        prevalence = float(prevalence) if prevalence is not None and np.isfinite(prevalence) and prevalence > 0 else None
        bundle.group_records[orpha] = {
            "name": row.name,
            "synonyms": [s for s in (row.synonyms if row.synonyms is not None else []) if s and s != row.name],
            "description": row.description or "",
            "phenotypes": _mean_weights([r["phenotypes"] for r in member_records]),
            "genes": _mean_weights([r["genes"] for r in member_records]),
            "inheritance": sorted({v for r in member_records for v in r["inheritance"]}),
            "onset": sorted({v for r in member_records for v in r["onset"]}),
            "prevalence": prevalence,
            "drugs": {},
            "ontology_parents": sorted(ontology_parents),
        }
        v1 = {
            column: frozenset().union(*(bundle.v1_sets[m][column] for m in members))
            for column in ("hpo_ids", "gene_symbols", "category_ids", "body_system_ids", "inheritance", "onset", "approved_drug_ids")
        }
        v1["prevalence"] = prevalence
        bundle.group_v1[orpha] = v1
        bundle.group_members[orpha] = frozenset(members)
        top = list(row.preferential_parent_names) if row.preferential_parent_names is not None else []
        rows.append(
            {
                "orpha_id": orpha,
                "name": row.name,
                "disorder_type": row.disorder_type,
                "disorder_group": row.disorder_group,
                "top_category": top[0] if top else "Unclassified",
                "expert_url": row.expert_url,
                "mondo_ids": ";".join(mondo_ids),
                "split": assign_split(orpha),
                "level": "group",
            }
        )
    bundle.group_meta = pd.DataFrame(rows).set_index("orpha_id").sort_index()


def attach_groups(bundle: Bundle, group_ids: Iterable[str]) -> list[str]:
    """Add the given Orphanet groups to the node set; returns the groups added."""

    added = [g for g in sorted(set(group_ids)) if g in bundle.group_records and g not in bundle.records]
    for g in added:
        bundle.records[g] = bundle.group_records[g]
        bundle.v1_sets[g] = bundle.group_v1[g]
    if added:
        bundle.meta = pd.concat([bundle.meta, bundle.group_meta.loc[added]]).sort_index()
    return added


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


def _load_genes(cohort_ids: set[str]) -> tuple[dict[str, dict[str, float]], dict[str, set[str]]]:
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
    return dict(genes), dict(causal)


def empty_record(name: str = "") -> dict[str, Any]:
    return {
        "name": name,
        "synonyms": [],
        "description": "",
        "phenotypes": {},
        "genes": {},
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
    for drug, weight in weighted(document.get("drugs", []), DESIGNATED_DRUG_WEIGHT).items():
        identifier = knowledge.drug_identifier(drug)
        if identifier is None:
            warnings.append(f"Ignored drug {drug!r}: name too short to identify")
            continue
        if identifier.startswith("NAME:"):
            warnings.append(f"Drug {drug!r} has no unique ChEMBL match; kept by name, so it has no known targets")
        record["drugs"][identifier] = max(weight, record["drugs"].get(identifier, 0.0))
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
