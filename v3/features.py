"""Pair features: static modality similarities and date-dependent features.

Static similarities ``S`` (cosine per modality) and availability ``A`` come
from the undated snapshot annotations.  ``TemporalFeatures`` adds what was
knowable at a snapshot date: designation history, the relation graph, and
the mechanisms (targets, target pathways) of drugs designated so far.  Every
feature is symmetric in the two diseases.

All block computations are "query rows x every node", so the same code
serves evaluation (rank the gallery), training (gather sampled columns) and
new diseases (query rows built from a user record instead of a node).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np
import scipy.sparse as sp

from modalities import MODALITIES
from world import Snapshot, World


def _normalize_rows(matrix: sp.csr_matrix) -> sp.csr_matrix:
    matrix = matrix.tocsr().astype(np.float32)
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return sp.csr_matrix(sp.diags(1.0 / norms) @ matrix, dtype=np.float32)


def _dense(x) -> np.ndarray:
    return x.toarray() if sp.issparse(x) else np.asarray(x)


class StaticSimilarity:
    def __init__(self, matrices: dict[str, sp.csr_matrix], available: np.ndarray):
        self.modalities = tuple(MODALITIES)
        self.matrices = {m: matrices[m].tocsr() for m in self.modalities}
        self.transposed = {m: self.matrices[m].T.tocsr() for m in self.modalities}
        self.available = available

    @classmethod
    def from_world(cls, world: World) -> "StaticSimilarity":
        return cls(world.matrices, world.available)

    def mask(self, masked: Sequence[str]) -> np.ndarray:
        return np.array([m in masked for m in self.modalities], dtype=bool)

    def block(
        self,
        rows: Optional[np.ndarray] = None,
        masked: Sequence[str] = (),
        query_matrices: Optional[dict[str, sp.csr_matrix]] = None,
        query_available: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """(S, A) of query rows (node indices, or new diseases) against every node."""

        if query_matrices is None:
            query_matrices = {m: self.matrices[m][rows] for m in self.modalities}
            query_available = self.available[rows]
        mask = self.mask(masked)
        nq = query_available.shape[0]
        n = self.available.shape[0]
        S = np.zeros((nq, n, len(self.modalities)), dtype=np.float32)
        for j, m in enumerate(self.modalities):
            if not mask[j]:
                S[:, :, j] = _dense(query_matrices[m] @ self.transposed[m])
        A = query_available[:, None, :] & self.available[None, :, :]
        A[:, :, mask] = False
        S[~A] = 0.0
        return S, A

    def pairs(self, a: np.ndarray, b: np.ndarray, masked: Sequence[str] = ()) -> tuple[np.ndarray, np.ndarray]:
        mask = self.mask(masked)
        S = np.zeros((len(a), len(self.modalities)), dtype=np.float32)
        step = 50_000
        for j, m in enumerate(self.modalities):
            if mask[j]:
                continue
            X = self.matrices[m]
            for start in range(0, len(a), step):
                sl = slice(start, start + step)
                S[sl, j] = np.asarray(X[a[sl]].multiply(X[b[sl]]).sum(axis=1)).ravel()
        A = self.available[a] & self.available[b]
        A[:, mask] = False
        S[~A] = 0.0
        return S, A


# --------------------------------------------------------------------------- temporal features

HISTORY_FEATURES = (
    "designations_min",
    "designations_max",
    "designations_sum",
    "degree_sum",
    "years_designated_min",
    "years_designated_max",
    "degree_min",
    "degree_max",
    "shared_drugs",
)
MECHANISM_FEATURES = ("target_cosine", "target_pathway_cosine", "gene_target", "pathway_target_pathway")
GRAPH_FEATURES = ("common_neighbors", "adamic_adar", "neighbor_jaccard", "neighbor_static_max", "neighbor_static_min")
NODE_FEATURES = ("groups_in_pair", "group_size_max", "oncology_in_pair", "same_category")
STATIC_LOGITS = ("static_logit", "neural_logit", "linear_logit")
STATIC_FEATURES = STATIC_LOGITS + tuple(f"sim_{m}" for m in MODALITIES) + ("static_available",)
ALL_FEATURES = STATIC_FEATURES + HISTORY_FEATURES + MECHANISM_FEATURES + GRAPH_FEATURES + NODE_FEATURES
FEATURE_GROUPS = {
    "static": STATIC_FEATURES + ("neighbor_static_max", "neighbor_static_min"),
    "history": HISTORY_FEATURES,
    "mechanism": MECHANISM_FEATURES,
    "graph": GRAPH_FEATURES,
    "node": NODE_FEATURES,
}

# (rows, S, A) -> {"static_logit", "neural_logit", "linear_logit"} blocks.
StaticScorer = Callable[[np.ndarray, np.ndarray, np.ndarray], dict[str, np.ndarray]]


def _vocabulary_matrix(rows: list[dict[str, float]], vocabulary: dict[str, int], n_cols: int) -> sp.csr_matrix:
    r, c, v = [], [], []
    for i, items in enumerate(rows):
        for item, weight in items.items():
            j = vocabulary.get(item)
            if j is not None:
                r.append(i)
                c.append(j)
                v.append(weight)
    return sp.csr_matrix((v, (r, c)), shape=(len(rows), n_cols), dtype=np.float32)


def _idf(matrix: sp.csr_matrix) -> np.ndarray:
    annotated = max(int((matrix.getnnz(axis=1) > 0).sum()), 1)
    df = np.asarray((matrix > 0).sum(axis=0)).ravel()
    return np.log((annotated + 1) / (df + 1)).astype(np.float32) + 1.0


@dataclass
class QueryRows:
    """Inputs of the temporal features for the query side of a block:
    catalogue nodes (``index`` set) or new diseases (``index`` None)."""

    index: Optional[np.ndarray]
    log_designations: np.ndarray
    years: np.ndarray
    degree: np.ndarray
    is_group: np.ndarray
    group_size: np.ndarray
    oncology: np.ndarray
    category: np.ndarray
    drugs: sp.csr_matrix
    targets: sp.csr_matrix
    targets_binary: sp.csr_matrix
    genes_binary: sp.csr_matrix
    target_pathways: sp.csr_matrix
    target_pathways_binary: sp.csr_matrix
    gene_pathways_binary: sp.csr_matrix
    adjacency: sp.csr_matrix

    def __len__(self) -> int:
        return len(self.log_designations)


class TemporalFeatures:
    """Feature blocks for one snapshot date and one static model."""

    def __init__(
        self,
        world: World,
        snapshot: Snapshot,
        static: StaticSimilarity,
        static_scorer: Optional[StaticScorer],
        reference_year: Optional[float] = None,
    ):
        self.world = world
        self.snapshot = snapshot
        self.static = static
        self.static_scorer = static_scorer
        self.knowledge = world.bundle.knowledge

        drug_rows = [snapshot.drugs.get(d, {}) for d in world.ids]
        self.drug_vocab = {d: i for i, d in enumerate(sorted({x for row in drug_rows for x in row}))}
        gene_rows = [dict.fromkeys(world.bundle.records[d].get("genes", {}), 1.0) for d in world.ids]
        target_rows = [self._targets_of(row) for row in drug_rows]
        genes = sorted({g for row in target_rows for g in row} | {g for row in gene_rows for g in row})
        self.gene_vocab = {g: i for i, g in enumerate(genes)}
        target_pathway_rows, gene_pathway_rows = self._pathways_of(target_rows), self._pathways_of(gene_rows)
        names = sorted({p for row in target_pathway_rows + gene_pathway_rows for p in row})
        self.pathway_vocab = {p: i for i, p in enumerate(names)}
        targets = self._matrix(target_rows, self.gene_vocab)
        target_pathways = self._matrix(target_pathway_rows, self.pathway_vocab)
        self.target_idf = _idf(targets)
        self.target_pathway_idf = _idf(target_pathways)

        if reference_year is None:
            reference_year = (
                snapshot.time.year + (snapshot.time.dayofyear - 1) / 365.25
                if snapshot.time is not None
                else float(np.nanmax(snapshot.first_designation)) + 1.0
            )
        self.reference_year = reference_year
        categories = {c: i for i, c in enumerate(sorted(set(world.top_category)))}
        self.categories = categories
        self.nodes = self._rows(
            index=np.arange(world.n),
            designations=snapshot.designations,
            first=snapshot.first_designation,
            adjacency=snapshot.adjacency.tocsr(),
            is_group=world.is_group,
            group_size=world.group_size,
            oncology=world.oncology,
            category=np.array([categories[c] for c in world.top_category]),
            drug_rows=drug_rows,
            gene_rows=gene_rows,
        )
        self.drugs_t = self.nodes.drugs.T.tocsr()
        self.targets_t = self.nodes.targets.T.tocsr()
        self.targets_binary_t = self.nodes.targets_binary.T.tocsr()
        self.genes_binary_t = self.nodes.genes_binary.T.tocsr()
        self.target_pathways_t = self.nodes.target_pathways.T.tocsr()
        self.target_pathways_binary_t = self.nodes.target_pathways_binary.T.tocsr()
        self.gene_pathways_binary_t = self.nodes.gene_pathways_binary.T.tocsr()
        self.adjacency = self.nodes.adjacency
        self.degree = self.nodes.degree
        self.adjacency_t = self.adjacency.T.tocsr()
        weights = 1.0 / np.log(2.0 + self.degree)
        self.adjacency_weighted_t = (sp.diags(weights) @ self.adjacency).T.tocsr()

    # ------------------------------------------------------------------ query rows

    def _targets_of(self, drugs: dict[str, float]) -> dict[str, float]:
        return {gene: 1.0 for drug in drugs for gene in self.knowledge.drug_targets.get(drug, ())}

    def _pathways_of(self, rows: list[dict[str, float]]) -> list[dict[str, float]]:
        return [{p: 1.0 for g in row for p in self.knowledge.gene_pathways.get(g, ())} for row in rows]

    @staticmethod
    def _matrix(rows: list[dict[str, float]], vocabulary: dict[str, int]) -> sp.csr_matrix:
        return _vocabulary_matrix(rows, vocabulary, max(len(vocabulary), 1))

    def _rows(
        self,
        index: Optional[np.ndarray],
        designations: np.ndarray,
        first: np.ndarray,
        adjacency: sp.csr_matrix,
        is_group: np.ndarray,
        group_size: np.ndarray,
        oncology: np.ndarray,
        category: np.ndarray,
        drug_rows: list[dict[str, float]],
        gene_rows: list[dict[str, float]],
    ) -> QueryRows:
        drugs = self._matrix(drug_rows, self.drug_vocab)
        drugs.data[:] = 1.0
        target_rows = [self._targets_of(row) for row in drug_rows]
        targets = self._matrix(target_rows, self.gene_vocab)
        target_pathways = self._matrix(self._pathways_of(target_rows), self.pathway_vocab)
        degree = np.asarray(adjacency.sum(axis=1)).ravel().astype(np.float32)
        return QueryRows(
            index=index,
            log_designations=np.log1p(designations).astype(np.float32),
            years=np.where(np.isnan(first), 0.0, self.reference_year - first).astype(np.float32),
            degree=degree,
            is_group=np.asarray(is_group, dtype=bool),
            group_size=np.asarray(group_size, dtype=np.float32),
            oncology=np.asarray(oncology, dtype=bool),
            category=np.asarray(category),
            drugs=drugs,
            targets=_normalize_rows(targets @ sp.diags(self.target_idf)),
            targets_binary=_normalize_rows(targets),
            genes_binary=_normalize_rows(self._matrix(gene_rows, self.gene_vocab)),
            target_pathways=_normalize_rows(target_pathways @ sp.diags(self.target_pathway_idf)),
            target_pathways_binary=_normalize_rows(target_pathways),
            gene_pathways_binary=_normalize_rows(self._matrix(self._pathways_of(gene_rows), self.pathway_vocab)),
            adjacency=adjacency,
        )

    def node_rows(self, rows: np.ndarray) -> QueryRows:
        n = self.nodes
        return QueryRows(
            index=rows,
            log_designations=n.log_designations[rows],
            years=n.years[rows],
            degree=n.degree[rows],
            is_group=n.is_group[rows],
            group_size=n.group_size[rows],
            oncology=n.oncology[rows],
            category=n.category[rows],
            drugs=n.drugs[rows],
            targets=n.targets[rows],
            targets_binary=n.targets_binary[rows],
            genes_binary=n.genes_binary[rows],
            target_pathways=n.target_pathways[rows],
            target_pathways_binary=n.target_pathways_binary[rows],
            gene_pathways_binary=n.gene_pathways_binary[rows],
            adjacency=n.adjacency[rows],
        )

    def new_rows(self, records: Sequence[dict[str, Any]], oncology: Optional[Sequence[bool]] = None) -> QueryRows:
        """Query rows for new diseases.  Their ``drugs`` are treated as
        designated now; they have no established relations yet."""

        k = len(records)
        drug_rows = [dict(r.get("drugs", {})) for r in records]
        return self._rows(
            index=None,
            designations=np.array([len(d) for d in drug_rows], dtype=np.float32),
            first=np.array([self.reference_year if d else np.nan for d in drug_rows], dtype=np.float64),
            adjacency=sp.csr_matrix((k, self.world.n), dtype=np.float32),
            is_group=np.zeros(k, dtype=bool),
            group_size=np.ones(k, dtype=np.float32),
            oncology=np.asarray(oncology if oncology is not None else [False] * k, dtype=bool),
            category=np.full(k, -1),
            drug_rows=drug_rows,
            gene_rows=[dict.fromkeys(r.get("genes", {}), 1.0) for r in records],
        )

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _pairwise(query: np.ndarray, gallery: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = query[:, None]
        g = gallery[None, :]
        return np.minimum(q, g), np.maximum(q, g)

    def _neighbor_static(self, q: QueryRows, logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Best static score between one disease and the other's known relatives.

        ``toward_gallery[q, g]`` = max over relatives r of g of logit(q, r);
        ``toward_query[q, g]``  = max over relatives r of q of logit(r, g)."""

        n = self.world.n
        adjacency = self.adjacency
        toward_gallery = np.full(logits.shape, np.nan, dtype=np.float32)
        nonempty = np.flatnonzero(np.diff(adjacency.indptr) > 0)
        if nonempty.size:
            gathered = logits[:, adjacency.indices]
            reduced = np.maximum.reduceat(gathered, adjacency.indptr[nonempty], axis=1)
            toward_gallery[:, nonempty] = reduced

        toward_query = np.full(logits.shape, np.nan, dtype=np.float32)
        qa = q.adjacency.tocsr()
        neighbor_lists = [qa.indices[qa.indptr[i] : qa.indptr[i + 1]] for i in range(len(q))]
        union = np.unique(np.concatenate(neighbor_lists)) if any(len(x) for x in neighbor_lists) else np.array([], int)
        if union.size and self.static_scorer is not None:
            position = {r: i for i, r in enumerate(union)}
            union_logits = np.empty((union.size, n), dtype=np.float32)
            for start in range(0, union.size, 32):
                chunk = union[start : start + 32]
                S, A = self.static.block(chunk)
                union_logits[start : start + 32] = self.static_scorer(chunk, S, A)["static_logit"]
            for i, neighbors in enumerate(neighbor_lists):
                if len(neighbors):
                    toward_query[i] = union_logits[[position[r] for r in neighbors]].max(axis=0)
        both = np.stack([toward_gallery, toward_query])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmax(both, axis=0), np.nanmin(both, axis=0)

    # ------------------------------------------------------------------ block

    def block(self, rows: np.ndarray, groups: Sequence[str] = tuple(FEATURE_GROUPS)) -> dict[str, np.ndarray]:
        """Named (len(rows) x N) feature matrices for catalogue nodes ``rows``."""

        parts = None
        S = A = None
        if "static" in groups or "graph" in groups:
            S, A = self.static.block(rows)
            parts = self.static_scorer(rows, S, A) if self.static_scorer is not None else {}
        return self.block_for(self.node_rows(rows), S, A, parts, groups)

    def block_for(
        self,
        q: QueryRows,
        S: Optional[np.ndarray],
        A: Optional[np.ndarray],
        parts: Optional[dict[str, np.ndarray]],
        groups: Sequence[str] = tuple(FEATURE_GROUPS),
    ) -> dict[str, np.ndarray]:
        """Feature block for any query rows given their static S, A and static logits."""

        out: dict[str, np.ndarray] = {}
        n = self.nodes
        logits = (parts or {}).get("static_logit")
        if "static" in groups:
            for name in STATIC_LOGITS:
                out[name] = (parts or {}).get(name, np.zeros(S.shape[:2], dtype=np.float32))
            for j, m in enumerate(MODALITIES):
                out[f"sim_{m}"] = np.where(A[:, :, j], S[:, :, j], np.nan).astype(np.float32)
            out["static_available"] = A.sum(axis=2).astype(np.float32)
        if "history" in groups:
            out["designations_min"], out["designations_max"] = self._pairwise(q.log_designations, n.log_designations)
            out["designations_sum"] = out["designations_min"] + out["designations_max"]
            out["years_designated_min"], out["years_designated_max"] = self._pairwise(q.years, n.years)
            q_log_degree, n_log_degree = np.log1p(q.degree), np.log1p(n.degree)
            out["degree_min"], out["degree_max"] = self._pairwise(q_log_degree, n_log_degree)
            out["degree_sum"] = out["degree_min"] + out["degree_max"]
            out["shared_drugs"] = _dense(q.drugs @ self.drugs_t).astype(np.float32)
        if "mechanism" in groups:
            out["target_cosine"] = _dense(q.targets @ self.targets_t)
            out["target_pathway_cosine"] = _dense(q.target_pathways @ self.target_pathways_t)
            out["gene_target"] = np.maximum(
                _dense(q.genes_binary @ self.targets_binary_t), _dense(q.targets_binary @ self.genes_binary_t)
            )
            out["pathway_target_pathway"] = np.maximum(
                _dense(q.gene_pathways_binary @ self.target_pathways_binary_t),
                _dense(q.target_pathways_binary @ self.gene_pathways_binary_t),
            )
        if "graph" in groups:
            common = _dense(q.adjacency @ self.adjacency_t).astype(np.float32)
            out["common_neighbors"] = common
            out["adamic_adar"] = _dense(q.adjacency @ self.adjacency_weighted_t).astype(np.float32)
            union = q.degree[:, None] + self.degree[None, :] - common
            out["neighbor_jaccard"] = np.divide(common, union, out=np.zeros_like(common), where=union > 0)
            if logits is not None:
                out["neighbor_static_max"], out["neighbor_static_min"] = self._neighbor_static(q, logits)
            else:
                out["neighbor_static_max"] = np.full(common.shape, np.nan, dtype=np.float32)
                out["neighbor_static_min"] = np.full(common.shape, np.nan, dtype=np.float32)
        if "node" in groups:
            out["groups_in_pair"] = (q.is_group[:, None].astype(np.float32) + n.is_group[None, :]).astype(np.float32)
            out["group_size_max"] = np.log(np.maximum(q.group_size[:, None], n.group_size[None, :]))
            out["oncology_in_pair"] = (q.oncology[:, None].astype(np.float32) + n.oncology[None, :]).astype(np.float32)
            out["same_category"] = (q.category[:, None] == n.category[None, :]).astype(np.float32)
        return {k: v.astype(np.float32, copy=False) for k, v in out.items()}


def stack(block: dict[str, np.ndarray], names: Sequence[str]) -> np.ndarray:
    """(rows, N, features) array in the given feature order."""

    return np.stack([block[name] for name in names], axis=-1)
