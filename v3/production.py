"""Production v3 model: everything refit on the full regulatory history.

Two scores per disease pair:

* ``similarity`` -- the static, drug-free relatedness (``HybridStatic``:
  neural multi-task ensemble + non-negative cosine fusion) trained on every
  relation established so far.  It defines the similarity graph and works for
  any disease, including a new one described only by phenotypes or text.
* ``forecast`` -- the stacker's score that the pair will be linked by a future
  orphan designation, combining the similarity with designation history,
  drug-mechanism overlap and the relation graph.  The stacker is trained on
  rolling origins up to the end of the data, exactly as in evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

from common import EVALUATION_DIR, MODEL_DIR, timed
from features import StaticSimilarity, TemporalFeatures
from modalities import MODALITIES
from neural import DEFAULT_CONFIG
from pipeline import (
    STACKER_VARIANTS,
    HybridStatic,
    Stacker,
    collect_stacker_data,
    fit_stackers,
    origins_for,
    stacker_importance,
    time_models,
)
from world import World, load_world

MODEL_PATH = MODEL_DIR / "production_model.joblib"
BACKGROUND_PAIRS = 100_000


def selected_stacker() -> str:
    path = EVALUATION_DIR / "selected_config.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8")).get("stacker", "v3_stacker")
    print("[v3] warning: no selected_config.json from run_evaluation.py; using v3_stacker")
    return "v3_stacker"


@dataclass
class ProductionModel:
    static: HybridStatic
    stacker: Stacker
    end_of_data: str
    n_nodes: int
    importance: dict[str, float]
    background_similarity: np.ndarray
    background_forecast: np.ndarray
    config: dict[str, Any]
    _world: Optional[World] = field(default=None, repr=False)
    _sim: Optional[StaticSimilarity] = field(default=None, repr=False)
    _features: Optional[TemporalFeatures] = field(default=None, repr=False)
    _masked_z: dict[str, list[np.ndarray]] = field(default_factory=dict, repr=False)
    _drug_names: Optional[dict[str, str]] = field(default=None, repr=False)

    def __getstate__(self):
        state = dict(self.__dict__)
        state.update(_world=None, _sim=None, _features=None, _masked_z={}, _drug_names=None)
        return state

    def drug_label(self, drug: str) -> str:
        """ChEMBL preferred name, else the most common name used in the designations."""

        label = self.world.bundle.knowledge.drug_labels.get(drug)
        if label:
            return label
        if self._drug_names is None:
            names = self.world.regulatory.designations.groupby("drug_id")["drug_name"]
            self._drug_names = names.agg(lambda s: s.mode().iat[0]).to_dict()
        return self._drug_names.get(drug, drug.removeprefix("NAME:"))

    # ------------------------------------------------------------------ runtime

    def attach(self, world: World) -> "ProductionModel":
        if world.n != self.n_nodes:
            raise RuntimeError("The cached world does not match the production model; rerun build_graph.py --refit.")
        self._world = world
        self._sim = StaticSimilarity.from_world(world)
        self._features = TemporalFeatures(world, world.snapshot(None), self._sim, self.static.parts)
        return self

    @property
    def world(self) -> World:
        return self._world

    @property
    def features(self) -> TemporalFeatures:
        return self._features

    def similarity_percentile(self, values: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.background_similarity, values, side="right") / self.background_similarity.size

    def forecast_percentile(self, values: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.background_forecast, values, side="right") / self.background_forecast.size

    # ------------------------------------------------------------------ scoring

    def score_nodes(self, rows: np.ndarray) -> dict[str, np.ndarray]:
        block = self.features.block(rows)
        block["forecast"] = self.stacker.score(block)
        return block

    def encode_record(self, record: dict[str, Any], hide: Sequence[str] = ()) -> tuple[dict[str, sp.csr_matrix], np.ndarray]:
        encoded = self.world.encoder.transform([record])
        available = np.array([[encoded[m].nnz > 0 and m not in hide for m in MODALITIES]])
        for j, m in enumerate(MODALITIES):
            if not available[0, j]:
                encoded[m] = sp.csr_matrix(encoded[m].shape, dtype=np.float32)
        return encoded, available

    def score_record(self, record: dict[str, Any], hide: Sequence[str] = (), oncology: bool = False) -> dict[str, Any]:
        """Similarity and forecast of one new disease against every node."""

        encoded, available = self.encode_record(record, hide)
        S, A = self._sim.block(query_matrices=encoded, query_available=available)
        z = self.static.neural.embeddings(encoded, available)
        parts = self.static.combine(self.static.neural.block_new(z, S, A), S, A)
        q = self.features.new_rows([record], oncology=[oncology])
        block = self.features.block_for(q, S, A, parts)
        block["forecast"] = self.stacker.score(block)
        present = [m for m, flag in zip(MODALITIES, available[0]) if flag]
        return {"block": block, "S": S[0], "A": A[0], "encoded": encoded, "z": z, "present": present, "query_rows": q}

    # ------------------------------------------------------------------ explanations

    def masked_embeddings(self, modality: str) -> list[np.ndarray]:
        if modality not in self._masked_z:
            self._masked_z[modality] = self.static.neural.embeddings(
                self.world.matrices, self.world.available, masked=(modality,)
            )
        return self._masked_z[modality]

    def neural_occlusion(
        self,
        a: np.ndarray,
        b: np.ndarray,
        S: np.ndarray,
        A: np.ndarray,
        query_z: Optional[list[np.ndarray]] = None,
        query_record: Optional[dict[str, Any]] = None,
        query_hide: Sequence[str] = (),
    ) -> np.ndarray:
        """(pairs x modalities) drop of the neural logit when a modality is withheld
        from both diseases.  ``a`` indexes nodes, or rows of ``query_z`` for a new disease."""

        z_nodes = [m.z for m in self.static.neural.models]
        za = [z[a] for z in (query_z if query_z is not None else z_nodes)]
        full = self.static.neural.pair_logits(za, [z[b] for z in z_nodes], S, A)
        drops = np.zeros((len(a), len(MODALITIES)), dtype=np.float32)
        for j, m in enumerate(MODALITIES):
            masked_b = [z[b] for z in self.masked_embeddings(m)]
            if query_record is not None:
                encoded, available = self.encode_record(query_record, hide=(*query_hide, m))
                masked_a = [z[a] for z in self.static.neural.embeddings(encoded, available)]
            else:
                masked_a = [z[a] for z in self.masked_embeddings(m)]
            S_m, A_m = S.copy(), A.copy()
            S_m[:, j], A_m[:, j] = 0.0, False
            drops[:, j] = full - self.static.neural.pair_logits(masked_a, masked_b, S_m, A_m)
        return drops

    def static_evidence(
        self,
        query_rows: dict[str, sp.csr_matrix],
        query_index: int,
        neighbor: int,
        S_row: np.ndarray,
        A_row: np.ndarray,
        occlusion: Optional[np.ndarray] = None,
        top_modalities: int = 4,
        top_features: int = 3,
    ) -> list[dict[str, Any]]:
        """Modalities behind a similarity: linear-fusion contribution, neural
        occlusion drop, and the shared items (phenotypes, genes, words...)."""

        linear = S_row * np.clip(self.static.linear.similarity_weights, 0, None) * A_row
        neural = occlusion if occlusion is not None else np.zeros(len(MODALITIES))
        combined = linear / self.static.linear_scale + np.clip(neural, 0, None) / self.static.neural_scale
        evidence = []
        for j in np.argsort(-combined)[:top_modalities]:
            if combined[j] <= 0:
                break
            m = MODALITIES[j]
            shared = self.world.encoder.shared_features(
                m, query_rows[m][query_index], self.world.matrices[m][neighbor], top=top_features
            )
            evidence.append(
                {
                    "modality": m,
                    "similarity": round(float(S_row[j]), 4),
                    "linear_contribution": round(float(linear[j]), 4),
                    "neural_occlusion": round(float(neural[j]), 4),
                    "shared": [label for label, _ in shared],
                }
            )
        return evidence

    def regulatory_evidence(self, q: Optional[int], neighbor: int, query_drugs: Optional[dict[str, float]] = None, query_genes=None) -> dict[str, Any]:
        """Drug-history evidence for a pair (q is a node index, or None for a new disease)."""

        world = self.world
        knowledge = world.bundle.knowledge
        snapshot = world.snapshot(None)
        drugs_q = query_drugs if q is None else snapshot.drugs.get(world.ids[q], {})
        drugs_n = snapshot.drugs.get(world.ids[neighbor], {})
        genes_q = set(query_genes if q is None else world.bundle.records[world.ids[q]].get("genes", {}))
        genes_n = set(world.bundle.records[world.ids[neighbor]].get("genes", {}))
        targets_q = {g for d in drugs_q for g in knowledge.drug_targets.get(d, ())}
        targets_n = {g for d in drugs_n for g in knowledge.drug_targets.get(d, ())}
        out: dict[str, Any] = {
            "designations": [len(drugs_q), len(drugs_n)],
            "shared_designated_drugs": sorted(self.drug_label(d) for d in set(drugs_q) & set(drugs_n))[:5],
            "shared_drug_targets": sorted(targets_q & targets_n)[:5],
            "gene_is_target_of_other": sorted((genes_q & targets_n) | (genes_n & targets_q))[:5],
        }
        if q is not None:
            adjacency = self.features.adjacency
            common = set(adjacency[q].indices) & set(adjacency[neighbor].indices)
            out["common_regulatory_relatives"] = [world.name(world.ids[c]) for c in sorted(common)][:5]
        return out


def fit_production(world: Optional[World] = None, config: Optional[dict[str, Any]] = None, seed: int = 0) -> ProductionModel:
    world = world or load_world()
    config = {**DEFAULT_CONFIG, **(config or {})}
    sim = StaticSimilarity.from_world(world)
    end_of_data = pd.Timestamp(world.regulatory.designations["date"].max())
    variant = selected_stacker()
    origins = origins_for(world, None, end_of_data)
    data = collect_stacker_data(world, sim, None, origins, config, seed)
    with timed(f"Fitting the production stacker ({variant}) on {len(data.y)} rows"):
        stacker = fit_stackers(data, {variant: STACKER_VARIANTS[variant]}, seed)[variant]
    models = time_models(world, sim, None, config=config)
    model = ProductionModel(
        static=models.static,
        stacker=stacker,
        end_of_data=str(end_of_data.date()),
        n_nodes=world.n,
        importance=stacker_importance(stacker, data, seed),
        background_similarity=np.zeros(0),
        background_forecast=np.zeros(0),
        config={"neural": config, "stacker": variant, "origins": [str(o.date()) for o in origins]},
    ).attach(world)
    rng = np.random.default_rng(seed)
    rows = rng.choice(world.n, 64, replace=False)
    with timed("Scoring random background pairs for percentiles"):
        block = model.score_nodes(rows)
        cols = rng.integers(0, world.n, (len(rows), BACKGROUND_PAIRS // len(rows)))
        pick = np.take_along_axis
        model.background_similarity = np.sort(pick(block["static_logit"], cols, axis=1).ravel())
        model.background_forecast = np.sort(pick(block["forecast"], cols, axis=1).ravel())
    return model


def save_model(model: ProductionModel) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)


def load_model() -> ProductionModel:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"{MODEL_PATH} not found. Run `python v3/build_graph.py` first.")
    model: ProductionModel = joblib.load(MODEL_PATH)
    return model.attach(load_world())
