"""Pair features: static modality similarities and date-dependent features.

Static similarities ``S`` (cosine per modality) and availability ``A`` come
from the undated snapshot annotations.  ``TemporalFeatures`` adds what was
knowable at a snapshot date: designation history, the relation graph, and
the mechanisms (targets, target pathways) of drugs designated so far.  Every
feature is symmetric in the two diseases.

All block computations are "query rows x every node", so the same code
serves evaluation (rank the gallery) and training (gather sampled columns).
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional, Sequence

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
STATIC_FEATURES = ("static_logit",) + tuple(f"sim_{m}" for m in MODALITIES) + ("static_available",)
ALL_FEATURES = STATIC_FEATURES + HISTORY_FEATURES + MECHANISM_FEATURES + GRAPH_FEATURES + NODE_FEATURES
FEATURE_GROUPS = {
    "static": STATIC_FEATURES + ("neighbor_static_max", "neighbor_static_min"),
    "history": HISTORY_FEATURES,
    "mechanism": MECHANISM_FEATURES,
    "graph": GRAPH_FEATURES,
    "node": NODE_FEATURES,
}

StaticScorer = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


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
        knowledge = world.bundle.knowledge
        n = world.n

        drug_rows = [snapshot.drugs.get(d, {}) for d in world.ids]
        drug_vocab = {d: i for i, d in enumerate(sorted({x for row in drug_rows for x in row}))}
        self.drugs = _vocabulary_matrix(drug_rows, drug_vocab, max(len(drug_vocab), 1))
        self.drugs.data[:] = 1.0

        target_rows = []
        for row in drug_rows:
            targets: dict[str, float] = {}
            for drug in row:
                for gene in knowledge.drug_targets.get(drug, ()):
                    targets[gene] = 1.0
            target_rows.append(targets)
        gene_rows = [dict.fromkeys(world.bundle.records[d].get("genes", {}), 1.0) for d in world.ids]
        genes = sorted({g for row in target_rows for g in row} | {g for row in gene_rows for g in row})
        gene_vocab = {g: i for i, g in enumerate(genes)}
        targets = _vocabulary_matrix(target_rows, gene_vocab, max(len(gene_vocab), 1))
        self.targets = _normalize_rows(targets @ sp.diags(_idf(targets)))
        self.targets_binary = _normalize_rows(targets)
        self.genes_binary = _normalize_rows(_vocabulary_matrix(gene_rows, gene_vocab, max(len(gene_vocab), 1)))

        def pathways(rows: list[dict[str, float]]) -> list[dict[str, float]]:
            return [{p: 1.0 for g in row for p in knowledge.gene_pathways.get(g, ())} for row in rows]

        target_pathway_rows, gene_pathway_rows = pathways(target_rows), pathways(gene_rows)
        names = sorted({p for row in target_pathway_rows + gene_pathway_rows for p in row})
        pathway_vocab = {p: i for i, p in enumerate(names)}
        tp = _vocabulary_matrix(target_pathway_rows, pathway_vocab, max(len(names), 1))
        self.target_pathways = _normalize_rows(tp @ sp.diags(_idf(tp)))
        self.target_pathways_binary = _normalize_rows(tp)
        self.gene_pathways_binary = _normalize_rows(_vocabulary_matrix(gene_pathway_rows, pathway_vocab, max(len(names), 1)))

        self.adjacency = snapshot.adjacency.tocsr()
        self.degree = snapshot.degree.astype(np.float32)
        weights = 1.0 / np.log(2.0 + self.degree)
        self.adjacency_weighted_t = (sp.diags(weights) @ self.adjacency).T.tocsr()
        self.adjacency_t = self.adjacency.T.tocsr()
        if reference_year is None:
            reference_year = (
                snapshot.time.year + (snapshot.time.dayofyear - 1) / 365.25
                if snapshot.time is not None
                else float(np.nanmax(snapshot.first_designation)) + 1.0
            )
        self.years = np.where(np.isnan(snapshot.first_designation), 0.0, reference_year - snapshot.first_designation)
        self.log_designations = np.log1p(snapshot.designations)
        self.log_degree = np.log1p(self.degree)
        categories = {c: i for i, c in enumerate(sorted(set(world.top_category)))}
        self.category = np.array([categories[c] for c in world.top_category])

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _pairwise(values: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = values[rows][:, None]
        g = values[None, :]
        return np.minimum(q, g), np.maximum(q, g)

    def _neighbor_static(self, rows: np.ndarray, logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
        neighbor_lists = [adjacency.indices[adjacency.indptr[q] : adjacency.indptr[q + 1]] for q in rows]
        union = np.unique(np.concatenate(neighbor_lists)) if any(len(x) for x in neighbor_lists) else np.array([], int)
        if union.size and self.static_scorer is not None:
            position = {r: i for i, r in enumerate(union)}
            union_logits = np.empty((union.size, n), dtype=np.float32)
            for start in range(0, union.size, 32):
                chunk = union[start : start + 32]
                S, A = self.static.block(chunk)
                union_logits[start : start + 32] = self.static_scorer(chunk, S, A)
            for i, neighbors in enumerate(neighbor_lists):
                if len(neighbors):
                    toward_query[i] = union_logits[[position[r] for r in neighbors]].max(axis=0)
        both = np.stack([toward_gallery, toward_query])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmax(both, axis=0), np.nanmin(both, axis=0)

    # ------------------------------------------------------------------ block

    def block(self, rows: np.ndarray, groups: Sequence[str] = tuple(FEATURE_GROUPS)) -> dict[str, np.ndarray]:
        """Named (len(rows) x N) feature matrices for the requested feature groups."""

        out: dict[str, np.ndarray] = {}
        logits = None
        if "static" in groups or "graph" in groups:
            S, A = self.static.block(rows)
            if self.static_scorer is not None:
                logits = self.static_scorer(rows, S, A)
            if "static" in groups:
                out["static_logit"] = logits if logits is not None else np.zeros(S.shape[:2], dtype=np.float32)
                for j, m in enumerate(MODALITIES):
                    out[f"sim_{m}"] = np.where(A[:, :, j], S[:, :, j], np.nan).astype(np.float32)
                out["static_available"] = A.sum(axis=2).astype(np.float32)
            del S, A
        if "history" in groups:
            out["designations_min"], out["designations_max"] = self._pairwise(self.log_designations, rows)
            out["designations_sum"] = out["designations_min"] + out["designations_max"]
            out["years_designated_min"], out["years_designated_max"] = self._pairwise(self.years, rows)
            out["degree_min"], out["degree_max"] = self._pairwise(self.log_degree, rows)
            out["degree_sum"] = out["degree_min"] + out["degree_max"]
            out["shared_drugs"] = _dense(self.drugs[rows] @ self.drugs.T).astype(np.float32)
        if "mechanism" in groups:
            out["target_cosine"] = _dense(self.targets[rows] @ self.targets.T)
            out["target_pathway_cosine"] = _dense(self.target_pathways[rows] @ self.target_pathways.T)
            out["gene_target"] = np.maximum(
                _dense(self.genes_binary[rows] @ self.targets_binary.T),
                _dense(self.targets_binary[rows] @ self.genes_binary.T),
            )
            out["pathway_target_pathway"] = np.maximum(
                _dense(self.gene_pathways_binary[rows] @ self.target_pathways_binary.T),
                _dense(self.target_pathways_binary[rows] @ self.gene_pathways_binary.T),
            )
        if "graph" in groups:
            common = _dense(self.adjacency[rows] @ self.adjacency_t).astype(np.float32)
            out["common_neighbors"] = common
            out["adamic_adar"] = _dense(self.adjacency[rows] @ self.adjacency_weighted_t).astype(np.float32)
            union = self.degree[rows][:, None] + self.degree[None, :] - common
            out["neighbor_jaccard"] = np.divide(common, union, out=np.zeros_like(common), where=union > 0)
            if logits is not None:
                out["neighbor_static_max"], out["neighbor_static_min"] = self._neighbor_static(rows, logits)
            else:
                out["neighbor_static_max"] = np.full(common.shape, np.nan, dtype=np.float32)
                out["neighbor_static_min"] = np.full(common.shape, np.nan, dtype=np.float32)
        if "node" in groups:
            w = self.world
            out["groups_in_pair"] = (w.is_group[rows][:, None].astype(np.float32) + w.is_group[None, :]).astype(np.float32)
            out["group_size_max"] = np.log(np.maximum(w.group_size[rows][:, None], w.group_size[None, :]))
            out["oncology_in_pair"] = (w.oncology[rows][:, None].astype(np.float32) + w.oncology[None, :]).astype(np.float32)
            out["same_category"] = (self.category[rows][:, None] == self.category[None, :]).astype(np.float32)
        return {k: v.astype(np.float32, copy=False) for k, v in out.items()}


def stack(block: dict[str, np.ndarray], names: Sequence[str]) -> np.ndarray:
    """(rows, N, features) array in the given feature order."""

    return np.stack([block[name] for name in names], axis=-1)


def log_odds(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p) - np.log1p(-p)
