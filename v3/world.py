"""The node set, static encodings, and what was known at any date.

Nodes are every cohort disease plus every eligible Orphanet group (2-200
member diseases), whether or not the group was ever designated: attaching
only designated groups would reveal which families later receive drugs.

A ``Snapshot`` is the state of regulatory knowledge just before a date: the
drugs designated for each node and the relations already established.  A
``Task`` asks which new relations appear in a time window, given the snapshot
at its start.  Candidates for a query exclude the query itself, its already
known relatives, and nodes nested with it (a group and its own members).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from common import CACHE_DIR, timed
from data_sources import Bundle, attach_groups, load_bundle
from modalities import MODALITIES, DiseaseEncoder, drug_vocabulary
from regulatory import RegulatoryData, load_regulatory

ONCOLOGY = "Rare neoplastic disease"
MAX_SIBLING_GROUP = 40
MAX_DISEASES_PER_CAUSAL_GENE = 20


@dataclass
class World:
    bundle: Bundle
    regulatory: RegulatoryData
    ids: list[str]
    index: dict[str, int]
    encoder: DiseaseEncoder
    matrices: dict[str, sp.csr_matrix]
    available: np.ndarray
    is_group: np.ndarray
    oncology: np.ndarray
    group_size: np.ndarray
    top_category: np.ndarray
    nested: sp.csr_matrix
    drug_words: frozenset[str]
    _snapshots: dict[pd.Timestamp, "Snapshot"] = field(default_factory=dict, repr=False)

    @property
    def n(self) -> int:
        return len(self.ids)

    def name(self, node: str) -> str:
        return self.bundle.records[node]["name"]

    def snapshot(self, time: Optional[pd.Timestamp]) -> "Snapshot":
        key = pd.Timestamp.max if time is None else pd.Timestamp(time)
        if key not in self._snapshots:
            self._snapshots[key] = build_snapshot(self, time)
        return self._snapshots[key]


@dataclass
class Snapshot:
    """Regulatory knowledge strictly before ``time`` (everything when None)."""

    time: Optional[pd.Timestamp]
    drugs: dict[str, dict[str, float]]
    relations: pd.DataFrame
    adjacency: sp.csr_matrix
    designations: np.ndarray
    first_designation: np.ndarray

    @property
    def warm(self) -> np.ndarray:
        return self.designations > 0

    @property
    def degree(self) -> np.ndarray:
        return np.asarray(self.adjacency.sum(axis=1)).ravel()


@dataclass
class Task:
    """Predict the relations established in ``[snapshot.time, end)``."""

    name: str
    snapshot: Snapshot
    end: Optional[pd.Timestamp]
    relations: pd.DataFrame
    positives: sp.csr_matrix

    def queries(self, gallery: np.ndarray, world: World) -> np.ndarray:
        """Nodes with at least one new relative among their gallery candidates."""

        in_gallery = np.zeros(world.n, dtype=bool)
        in_gallery[gallery] = True
        rows = np.asarray(self.positives[:, in_gallery].sum(axis=1)).ravel() > 0
        return np.flatnonzero(rows & in_gallery)

    def excluded(self, queries: np.ndarray, world: World) -> sp.csr_matrix:
        """Known relatives, nested nodes and the query itself (rows = queries)."""

        identity = sp.csr_matrix(
            (np.ones(len(queries), dtype=np.float32), (np.arange(len(queries)), queries)), shape=(len(queries), world.n)
        )
        return ((self.snapshot.adjacency[queries] + world.nested[queries] + identity) > 0).tocsr()


def pair_matrix(n: int, a: Sequence[int], b: Sequence[int]) -> sp.csr_matrix:
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    data = np.ones(2 * len(a), dtype=np.float32)
    matrix = sp.csr_matrix((data, (np.r_[a, b], np.r_[b, a])), shape=(n, n))
    matrix.data[:] = 1.0
    return matrix


def build_world(rebuild_regulatory: bool = False) -> World:
    bundle = load_bundle()
    regulatory = load_regulatory(bundle, rebuild=rebuild_regulatory)
    attach_groups(bundle, bundle.group_records)
    ids = sorted(bundle.records)
    index = {d: i for i, d in enumerate(ids)}
    records = [bundle.records[d] for d in ids]
    drug_names = set(regulatory.designations["drug_name"].dropna())
    labels = bundle.knowledge.drug_labels
    drug_names |= {labels[d] for d in regulatory.designations["drug_id"] if d in labels}
    drug_words = drug_vocabulary(drug_names, bundle.knowledge)
    with timed(f"Encoding {len(ids)} nodes ({len(drug_words)} drug words removed from descriptions)"):
        encoder = DiseaseEncoder(bundle.knowledge, MODALITIES, drug_words).fit(records, records)
        matrices = encoder.transform(records)
    available = np.stack([matrices[m].getnnz(axis=1) > 0 for m in MODALITIES], axis=1)
    meta = bundle.meta.loc[ids]
    is_group = (meta["level"] == "group").to_numpy()
    group_size = np.array([len(bundle.group_members.get(d, ())) or 1 for d in ids], dtype=np.float32)
    top_category = meta["top_category"].to_numpy()

    catalog = regulatory.catalog
    a, b = [], []
    for group in bundle.group_members:
        if group not in index:
            continue
        for node in catalog.descendants(group):
            if node in index and node != group:
                a.append(index[group])
                b.append(index[node])
    nested = pair_matrix(len(ids), a, b)
    return World(
        bundle=bundle,
        regulatory=regulatory,
        ids=ids,
        index=index,
        encoder=encoder,
        matrices=matrices,
        available=available,
        is_group=is_group,
        oncology=(top_category == ONCOLOGY),
        group_size=group_size,
        top_category=top_category,
        nested=nested,
        drug_words=drug_words,
    )


def load_world(rebuild: bool = False) -> World:
    """Build (or load the cached) world."""

    import pickle

    path = CACHE_DIR / "world.pkl"
    if path.is_file() and not rebuild:
        with path.open("rb") as handle:
            return pickle.load(handle)
    world = build_world(rebuild_regulatory=rebuild)
    world._snapshots = {}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(world, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return world


def build_snapshot(world: World, time: Optional[pd.Timestamp]) -> Snapshot:
    regulatory = world.regulatory
    relations = regulatory.relations if time is None else regulatory.known_before(time)
    drugs = {d: v for d, v in regulatory.drugs_before(time).items() if d in world.index}
    designations = np.zeros(world.n, dtype=np.float32)
    first = np.full(world.n, np.nan, dtype=np.float64)
    for disease, dated in regulatory.history.items():
        i = world.index.get(disease)
        if i is None:
            continue
        dates = [t for t in dated.values() if time is None or t < time]
        if dates:
            designations[i] = len(dates)
            first[i] = min(dates).year + (min(dates).dayofyear - 1) / 365.25
    adjacency = pair_matrix(
        world.n, [world.index[x] for x in relations["a"]], [world.index[x] for x in relations["b"]]
    )
    return Snapshot(
        time=time,
        drugs=drugs,
        relations=relations,
        adjacency=adjacency,
        designations=designations,
        first_designation=first,
    )


def make_task(world: World, name: str, start: pd.Timestamp, end: Optional[pd.Timestamp] = None) -> Task:
    snapshot = world.snapshot(start)
    relations = world.regulatory.new_after(start, end)
    positives = pair_matrix(
        world.n, [world.index[x] for x in relations["a"]], [world.index[x] for x in relations["b"]]
    )
    return Task(name=name, snapshot=snapshot, end=end, relations=relations, positives=positives)


# --------------------------------------------------------------------------- auxiliary relations


def sibling_pairs(world: World, max_pairs: int, rng: np.random.Generator) -> np.ndarray:
    """Cohort diseases sharing a direct Orphanet parent with at most 40 cohort children."""

    children: dict[str, list[int]] = defaultdict(list)
    for parent, child in world.bundle.labels.orphanet_edges:
        i = world.index.get(child)
        if i is not None and not world.is_group[i]:
            children[parent].append(i)
    pairs = set()
    for members in children.values():
        members = sorted(set(members))
        if 2 <= len(members) <= MAX_SIBLING_GROUP:
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    pairs.add((members[x], members[y]))
    return _cap(np.array(sorted(pairs), dtype=np.int64).reshape(-1, 2), max_pairs, rng)


def gene_pairs(world: World, max_pairs: int, rng: np.random.Generator) -> np.ndarray:
    """Cohort diseases sharing a causal gene (genes with at most 20 diseases)."""

    by_gene: dict[str, list[int]] = defaultdict(list)
    for disease, genes in world.bundle.labels.causal_genes.items():
        i = world.index.get(disease)
        if i is None or world.is_group[i]:
            continue
        for gene in genes:
            by_gene[gene].append(i)
    pairs = set()
    for members in by_gene.values():
        members = sorted(set(members))
        if 2 <= len(members) <= MAX_DISEASES_PER_CAUSAL_GENE:
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    pairs.add((members[x], members[y]))
    return _cap(np.array(sorted(pairs), dtype=np.int64).reshape(-1, 2), max_pairs, rng)


def _cap(pairs: np.ndarray, max_pairs: int, rng: np.random.Generator) -> np.ndarray:
    if len(pairs) > max_pairs:
        pairs = pairs[rng.choice(len(pairs), max_pairs, replace=False)]
    return pairs


def describe(world: World) -> dict[str, Any]:
    return {
        "nodes": world.n,
        "groups": int(world.is_group.sum()),
        "modalities": {m: int(world.available[:, j].sum()) for j, m in enumerate(MODALITIES)},
        "drug_words_removed": len(world.drug_words),
    }
