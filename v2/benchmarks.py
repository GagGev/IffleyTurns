"""Database-derived relatedness benchmarks.

Each benchmark is a set of curated disease-disease relations from one source:

* ``orphanet_siblings`` -- Orphanet experts placed both diseases directly under
  the same (fine-grained) classification group, or one is a subtype of the
  other (Orphadata product 3).
* ``shared_causal_gene`` -- Orphanet lists a common gene carrying
  disease-causing germline mutations for both (Orphadata product 6, assessed).
* ``shared_drug`` -- both are indications of the same drug in a phase 2, 3, or
  approved stage (Open Targets / ChEMBL clinical indications), restricted to
  specific drugs with at most ``MAX_DISEASES_PER_DRUG`` cohort indications and
  to pairs backed by distinct indication records.

To avoid circular evaluation, every benchmark *masks* the feature modalities
derived from its own label source; a model must recover the relation from
independent evidence.  Splits are by disease, so every validation/test query is
a disease that no encoder or model saw during fitting -- the "new disease"
setting.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from common import pair_key
from data_sources import Bundle

MAX_SIBLING_GROUP = 40
MAX_DISEASES_PER_DRUG = 20


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    title: str
    description: str
    masked: tuple[str, ...]
    v1_masked: tuple[str, ...]


BENCHMARK_SPECS = {
    "orphanet_siblings": BenchmarkSpec(
        name="orphanet_siblings",
        title="Orphanet classification siblings",
        description=(
            "Pairs placed directly under the same Orphanet classification group "
            f"(<= {MAX_SIBLING_GROUP} member diseases) or in a disorder-subtype relation (Orphadata product 3)."
        ),
        masked=("ontology", "name"),
        v1_masked=("classifications", "body_systems"),
    ),
    "shared_causal_gene": BenchmarkSpec(
        name="shared_causal_gene",
        title="Shared causal gene",
        description=(
            "Pairs with a common gene carrying disease-causing germline mutations, as assessed by "
            "Orphanet (Orphadata product 6)."
        ),
        masked=("gene", "pathway", "ot_gene"),
        v1_masked=("genes",),
    ),
    "shared_drug": BenchmarkSpec(
        name="shared_drug",
        title="Shared trial or approved drug",
        description=(
            "Pairs that are both indications of the same drug at phase 2 or later (Open Targets / ChEMBL "
            f"clinical indications), using drugs with <= {MAX_DISEASES_PER_DRUG} cohort indications and "
            "distinct indication records."
        ),
        masked=("drug", "drug_target", "ot_gene"),
        # v1 gene sets mix in Open Targets associations, which include ChEMBL drug evidence.
        v1_masked=("approved_drugs", "genes"),
    ),
}


@dataclass
class Relation:
    spec: BenchmarkSpec
    eligible: frozenset[str]
    neighbors: dict[str, set[str]]
    evidence: dict[tuple[str, str], str]
    hard_neighbors: dict[str, set[str]] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.spec.name

    def n_pairs(self) -> int:
        return len(self.evidence)

    def pairs_within(self, members: set[str]) -> list[tuple[str, str]]:
        return [pair for pair in self.evidence if pair[0] in members and pair[1] in members]

    def queries(self, split_ids: Iterable[str]) -> list[str]:
        return sorted(d for d in split_ids if d in self.eligible and self.neighbors.get(d))


def _add_pair(neighbors, evidence, a, b, reason):
    if a == b:
        return
    neighbors[a].add(b)
    neighbors[b].add(a)
    key = pair_key(a, b)
    if key in evidence:
        if reason not in evidence[key]:
            evidence[key] = evidence[key] + "; " + reason
    else:
        evidence[key] = reason


class OrphanetIndex:
    """Orphanet tree helpers shared by the sibling benchmark, strata, and controls."""

    def __init__(self, bundle: Bundle):
        cohort = set(bundle.records)
        knowledge = bundle.knowledge
        self.children: dict[str, set[str]] = defaultdict(set)
        self.parents: dict[str, set[str]] = defaultdict(set)
        for parent, child in bundle.labels.orphanet_edges:
            self.children[parent].add(child)
            self.parents[child].add(parent)
        heads = knowledge.orphanet_heads
        self.ancestors: dict[str, frozenset[str]] = {}
        for disease in cohort:
            ancestors = {a for a in knowledge.ontology_ancestors(disease) if a.startswith("ORPHA:")}
            ancestors.discard(disease)
            self.ancestors[disease] = frozenset(ancestors - heads)
        self.cohort = cohort

    def close(self, a: str, b: str) -> bool:
        """True if a and b share a direct Orphanet parent or one descends from the other."""

        if self.parents.get(a, set()) & self.parents.get(b, set()):
            return True
        return a in self.ancestors.get(b, ()) or b in self.ancestors.get(a, ())


def build_relations(bundle: Bundle) -> dict[str, Relation]:
    cohort = set(bundle.records)
    labels = bundle.labels
    names = bundle.knowledge.ontology_labels
    orphanet = OrphanetIndex(bundle)
    relations: dict[str, Relation] = {}

    neighbors: dict[str, set[str]] = defaultdict(set)
    evidence: dict[tuple[str, str], str] = {}
    eligible = {d for d in cohort if orphanet.parents.get(d)}
    for parent, children in orphanet.children.items():
        members = sorted(c for c in children if c in cohort)
        if parent in cohort:
            for child in members:
                _add_pair(neighbors, evidence, parent, child, f"subtype of {names.get(parent, parent)} ({parent})")
        if 2 <= len(members) <= MAX_SIBLING_GROUP:
            reason = f"siblings under {names.get(parent, parent)} ({parent})"
            for a, b in itertools.combinations(members, 2):
                _add_pair(neighbors, evidence, a, b, reason)
    relations["orphanet_siblings"] = Relation(
        BENCHMARK_SPECS["orphanet_siblings"], frozenset(eligible), dict(neighbors), evidence
    )

    neighbors, evidence = defaultdict(set), {}
    by_gene: dict[str, list[str]] = defaultdict(list)
    for disease, genes in labels.causal_genes.items():
        if disease in cohort:
            for gene in genes:
                by_gene[gene].append(disease)
    for gene, diseases in by_gene.items():
        for a, b in itertools.combinations(sorted(diseases), 2):
            _add_pair(neighbors, evidence, a, b, f"causal gene {gene}")
    eligible = {d for d, genes in labels.causal_genes.items() if d in cohort and genes}
    relations["shared_causal_gene"] = Relation(
        BENCHMARK_SPECS["shared_causal_gene"], frozenset(eligible), dict(neighbors), evidence
    )

    neighbors, evidence = defaultdict(set), {}
    by_drug: dict[str, list[str]] = defaultdict(list)
    for disease, drugs in labels.trial_drugs.items():
        if disease in cohort:
            for drug in drugs:
                by_drug[drug].append(disease)
    specific = {drug: ds for drug, ds in by_drug.items() if len(ds) <= MAX_DISEASES_PER_DRUG}
    drug_labels = bundle.knowledge.drug_labels
    for drug, diseases in specific.items():
        for a, b in itertools.combinations(sorted(diseases), 2):
            # Identical indication records mean one Open Targets disease was
            # cross-referenced to both Orphanet entities, not two indications.
            if labels.trial_drugs[a][drug] == labels.trial_drugs[b][drug]:
                continue
            _add_pair(neighbors, evidence, a, b, f"drug {drug_labels.get(drug, drug)} ({drug})")
    eligible = {d for ds in specific.values() for d in ds}
    relations["shared_drug"] = Relation(
        BENCHMARK_SPECS["shared_drug"], frozenset(eligible), dict(neighbors), evidence
    )

    for relation in relations.values():
        relation.neighbors = {d: n & relation.eligible for d, n in relation.neighbors.items() if d in relation.eligible}
        if relation.name != "orphanet_siblings":
            relation.hard_neighbors = {
                d: {n for n in ns if not orphanet.close(d, n)} for d, ns in relation.neighbors.items()
            }
    return relations


def controls_mask(bundle: Bundle, orphanet: OrphanetIndex, query: str, gallery: list[str]) -> np.ndarray:
    """Clearly unrelated candidates: different top-level Orphanet category, no
    shared Orphanet group below the classification heads, and no shared curated
    gene or drug."""

    meta_top = bundle.meta["top_category"]
    q_top = meta_top.get(query)
    q_anc = orphanet.ancestors.get(query, frozenset())
    q_genes = set(bundle.records[query]["genes"]) | set(bundle.labels.causal_genes.get(query, ()))
    q_drugs = set(bundle.records[query]["drugs"])
    mask = np.zeros(len(gallery), dtype=bool)
    for i, candidate in enumerate(gallery):
        if candidate == query or meta_top.get(candidate) == q_top:
            continue
        if q_anc & orphanet.ancestors.get(candidate, frozenset()):
            continue
        record = bundle.records[candidate]
        if q_genes & set(record["genes"]) or q_drugs & set(record["drugs"]):
            continue
        mask[i] = True
    return mask


def sample_training_pairs(
    relation: Relation,
    members: set[str],
    negatives_per_positive: int,
    rng: np.random.Generator,
) -> tuple[list[tuple[str, str]], np.ndarray]:
    """Positive pairs inside ``members`` plus uniformly sampled unrelated pairs."""

    eligible = sorted(relation.eligible & members)
    positives = relation.pairs_within(set(eligible))
    positive_keys = set(positives)
    n_negative = negatives_per_positive * len(positives)
    negatives: set[tuple[str, str]] = set()
    n = len(eligible)
    while len(negatives) < n_negative and n > 1:
        a = rng.integers(0, n, size=2 * (n_negative - len(negatives)) + 16)
        b = rng.integers(0, n, size=a.size)
        for i, j in zip(a, b):
            if i == j:
                continue
            key = pair_key(eligible[i], eligible[j])
            if key not in positive_keys:
                negatives.add(key)
                if len(negatives) >= n_negative:
                    break
    pairs = positives + sorted(negatives)
    labels = np.r_[np.ones(len(positives)), np.zeros(len(negatives))].astype(np.int8)
    return pairs, labels
