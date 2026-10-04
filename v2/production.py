"""Production similarity model: encoders and fusion fitted on the full catalogue.

The model is the multi-task logistic fusion selected by ``run_evaluation.py``.
Each modality's weight is learned only from benchmarks whose labels come from
a different source.  It is saved with everything needed to place a brand-new
disease into the graph: fitted encoders, reference knowledge, the encoded
catalogue, the training pairs (to refit the fusion to a new disease's
available modalities), and random background pairs for percentiles.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import joblib
import numpy as np
import scipy.sparse as sp

from benchmarks import build_relations
from common import EVALUATION_DIR, MODEL_DIR, pair_key, timed
from data_sources import Bundle, Knowledge
from modalities import DiseaseEncoder
from scoring import LinearFusion, PatternFusion, SimilarityEngine, TrainingSet, build_training_set, fit_logistic

DEFAULT_CONFIG = {
    "logistic_params": {"C": 0.3, "use_availability": True, "nonnegative": True},
    "new_disease_fusion": "pattern",
    "negatives_per_positive": 10,
    "seed": 0,
}
MODEL_PATH = MODEL_DIR / "production_model.joblib"
BACKGROUND_PAIRS = 200_000


@dataclass
class ProductionModel:
    encoder: DiseaseEncoder
    fusion: LinearFusion
    knowledge: Knowledge
    ids: list[str]
    names: list[str]
    top_categories: list[str]
    matrices: dict[str, sp.csr_matrix]
    relation_evidence: dict[str, dict[tuple[str, str], str]]
    training: TrainingSet
    background_S: np.ndarray
    background_A: np.ndarray
    config: dict[str, Any]
    _engine: Optional[SimilarityEngine] = field(default=None, repr=False)
    _pattern: Optional[PatternFusion] = field(default=None, repr=False)
    _backgrounds: dict[int, np.ndarray] = field(default_factory=dict, repr=False)

    def __getstate__(self):
        state = dict(self.__dict__)
        state.update(_engine=None, _pattern=None, _backgrounds={})
        return state

    @property
    def engine(self) -> SimilarityEngine:
        if self._engine is None:
            self._engine = SimilarityEngine(self.matrices, self.ids)
        return self._engine

    @property
    def pattern(self) -> PatternFusion:
        if self._pattern is None:
            self._pattern = PatternFusion(self.training, self.fusion)
        return self._pattern

    def fusion_for(self, present: Sequence[str]) -> LinearFusion:
        """Fusion used to place a new disease annotated with the ``present`` modalities."""

        if self.config.get("new_disease_fusion", "pattern") != "pattern":
            return self.fusion
        return self.pattern.submodel([m for m in self.engine.modalities if m not in present])

    def extend(self, ids: Sequence[str], names: Sequence[str], records: Sequence[dict[str, Any]]) -> None:
        """Add (user-supplied) diseases to the gallery so later placements can find them."""

        if not ids:
            return
        encoded = self.encoder.transform(list(records))
        self.matrices = {m: sp.vstack([self.matrices[m], encoded[m]]).tocsr() for m in self.matrices}
        self.ids = list(self.ids) + list(ids)
        self.names = list(self.names) + list(names)
        self.top_categories = list(self.top_categories) + ["User-added disease"] * len(ids)
        self._engine = None

    def percentile(self, logits: np.ndarray, fusion: Optional[LinearFusion] = None) -> np.ndarray:
        """Share of random catalogue pairs that ``fusion`` scores below each logit."""

        fusion = fusion or self.fusion
        key = id(fusion)
        if key not in self._backgrounds:
            self._backgrounds[key] = np.sort(fusion.score(self.background_S[None], self.background_A[None])[0])
        background = self._backgrounds[key]
        return np.searchsorted(background, logits, side="right") / background.size

    def score_record(self, record: dict[str, Any], hide: Sequence[str] = ()) -> dict[str, Any]:
        """Score one (new) disease against the whole gallery with the fusion fitted to its modalities."""

        encoded = self.encoder.transform([record])
        modalities = self.engine.modalities
        available = np.array([[encoded[m].nnz > 0 and m not in hide for m in modalities]])
        S, A = self.engine.block(encoded, available, self.engine.gallery(self.ids), masked=hide)
        present = [m for m, flag in zip(modalities, available[0]) if flag]
        fusion = self.fusion_for(present)
        return {"logits": fusion.score(S, A)[0], "S": S[0], "A": A[0], "encoded": encoded, "present": present, "fusion": fusion}

    def explain(
        self,
        query_rows: dict[str, sp.csr_matrix],
        query_index: int,
        gallery_index: int,
        S_row: np.ndarray,
        A_row: np.ndarray,
        top_modalities: int = 4,
        top_features: int = 3,
        fusion: Optional[LinearFusion] = None,
    ) -> list[dict[str, Any]]:
        """Per-modality similarity contributions with the shared features behind them."""

        contributions = (fusion or self.fusion).similarity_contributions(S_row)
        order = np.argsort(-contributions)
        explanation = []
        for j in order[:top_modalities]:
            if contributions[j] <= 0:
                break
            modality = self.engine.modalities[j]
            shared = self.encoder.shared_features(
                modality,
                query_rows[modality][query_index],
                self.matrices[modality][gallery_index],
                top=top_features,
            )
            explanation.append(
                {
                    "modality": modality,
                    "similarity": round(float(S_row[j]), 4),
                    "contribution": round(float(contributions[j]), 4),
                    "shared": [label for label, _ in shared],
                }
            )
        return explanation

    def known_relations(self, a: str, b: str) -> dict[str, str]:
        key = pair_key(a, b)
        return {name: evidence[key] for name, evidence in self.relation_evidence.items() if key in evidence}

    def save(self) -> None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, MODEL_PATH, compress=3)


def load_model() -> ProductionModel:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"{MODEL_PATH} not found. Run `python v2/build_graph.py` first.")
    return joblib.load(MODEL_PATH)


def selected_config() -> dict[str, Any]:
    path = EVALUATION_DIR / "selected_config.json"
    if path.is_file():
        return {**DEFAULT_CONFIG, **json.loads(path.read_text(encoding="utf-8"))}
    print("[v2] warning: no selected_config.json from run_evaluation.py; using defaults")
    return dict(DEFAULT_CONFIG)


def fit_production(bundle: Bundle, config: Optional[dict[str, Any]] = None) -> ProductionModel:
    """Fit encoders and the selected fusion on every catalogue disease."""

    config = config or selected_config()
    ids = sorted(bundle.records)
    records = [bundle.records[d] for d in ids]
    with timed("Fitting production encoders on the full catalogue"):
        encoder = DiseaseEncoder(bundle.knowledge).fit(records, records)
        matrices = encoder.transform(records)
    engine = SimilarityEngine(matrices, ids)
    relations = build_relations(bundle)
    with timed("Fitting production fusion (masked multi-task logistic regression)"):
        data = build_training_set(
            engine, list(relations.values()), set(ids), config["negatives_per_positive"], config.get("seed", 0)
        )
        params = config["logistic_params"]
        fusion = fit_logistic(data, engine.modalities, params["C"], params["use_availability"], params.get("nonnegative", True))
    rng = np.random.default_rng(config.get("seed", 0))
    a = rng.integers(0, len(ids), BACKGROUND_PAIRS)
    b = rng.integers(0, len(ids), BACKGROUND_PAIRS)
    keep = a != b
    background_S, background_A = engine.pair_features([ids[i] for i in a[keep]], [ids[i] for i in b[keep]])
    return ProductionModel(
        encoder=encoder,
        fusion=fusion,
        knowledge=bundle.knowledge,
        ids=ids,
        names=[bundle.records[d]["name"] for d in ids],
        top_categories=[bundle.meta.at[d, "top_category"] for d in ids],
        matrices=matrices,
        relation_evidence={name: dict(r.evidence) for name, r in relations.items()},
        training=data,
        background_S=background_S,
        background_A=background_A,
        config=config,
    )
