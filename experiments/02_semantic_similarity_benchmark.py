"""Benchmark compact semantic disease-pair features against the current model.

This experiment is deliberately isolated: it reads the project's generated
feature and split files, but every artifact it creates is written below the
``experiments`` directory.

The primary benchmark keeps the existing paper-averaged target and current
train/validation/test pair split so that the comparison is apples-to-apples.
It compares:

* the current 249-feature hashed representation;
* compact structured features with HPO ancestor semantics and curated genes;
* the compact representation plus TF-IDF similarity of disease descriptions;
* a combination of the current and improved representations.

A secondary retrieval benchmark asks whether each high-similarity literature
edge ranks above deterministic, body-system-matched unobserved controls.  Those
controls are not asserted to be biologically unrelated; this is explicitly a
ranking stress test rather than a source of clinical ground truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

sys.dont_write_bytecode = True

import joblib
import numpy as np
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import DiseaseDistanceEvaluator  # noqa: E402
from models.evaluation import regression_metrics  # noqa: E402
from models.linear_classifier import (  # noqa: E402
    FEATURE_NAMES as CURRENT_FEATURE_NAMES,
    extract_pair_features,
    load_pair_targets,
)


DEFAULT_FEATURE_DIR = PROJECT_ROOT / ".data" / "features"
DEFAULT_SPLIT_DIR = PROJECT_ROOT / ".data" / "splits"
DEFAULT_HPO_OBO = PROJECT_ROOT / ".data" / "databases" / "hpo" / "hp.obo"
DEFAULT_OUTPUT_DIR = EXPERIMENT_ROOT / "artifacts" / "semantic_similarity_v1"
SPLIT_NAMES = ("train", "validation", "test")
TRUSTED_GENE_SOURCES = frozenset(
    {
        "orphadata_product6",
        "hpo_genes_to_disease",
        "clingen_gene_validity",
        "clinvar_gene_condition",
    }
)
OPEN_TARGETS_SOURCE = "opentargets_association_overall_direct"
OPEN_TARGETS_HIGH_CONFIDENCE = 0.5
SET_STAT_SUFFIXES = (
    "both_available",
    "jaccard",
    "overlap_coefficient",
    "size_balance",
    "log1p_shared",
    "log1p_union",
)
PRIMARY_IMPROVED_VARIANTS = ("improved_structured", "improved_full")


@dataclass(frozen=True)
class PairRecord:
    """One paper-averaged unordered disease pair."""

    split: str
    pair_id: str
    orpha_id_a: str
    orpha_id_b: str
    target: float
    paper_count: int
    relationships: str


@dataclass
class FeatureContext:
    """In-memory disease features used by the compact representation."""

    diseases: Dict[str, Dict[str, Any]]
    hpo_ancestors: Callable[[str], frozenset[str]]
    hpo_ic: Dict[str, float]
    hpo_profiles: Dict[str, Dict[str, float]]
    trusted_genes: Dict[str, set[str]]
    high_confidence_ot_genes: Dict[str, set[str]]
    description_matrix: sparse.csr_matrix
    description_index: Dict[str, int]


@dataclass(frozen=True)
class Candidate:
    """A deterministic model candidate selected on validation RMSE."""

    name: str
    parameters: Dict[str, Any]


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(
                value,
                stream,
                indent=2,
                sort_keys=True,
                default=_json_default,
            )
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_output_is_isolated(output_dir: Path) -> Path:
    resolved = output_dir.expanduser().resolve()
    try:
        resolved.relative_to(EXPERIMENT_ROOT)
    except ValueError as error:
        raise ValueError(
            f"Output directory must be inside {EXPERIMENT_ROOT}; got {resolved}."
        ) from error
    return resolved


def load_records(split_dir: Path) -> Dict[str, list[PairRecord]]:
    """Load the current pair split with production-equivalent aggregation."""

    result: Dict[str, list[PairRecord]] = {}
    for split_name in SPLIT_NAMES:
        targets = load_pair_targets(
            split_dir / f"{split_name}.csv",
            split_name,
        )
        records: list[PairRecord] = []
        for target in targets:
            parts = str(target["pair_id"]).split("|")
            if len(parts) != 2:
                raise ValueError(f"Invalid pair ID {target['pair_id']!r}.")
            records.append(
                PairRecord(
                    split=split_name,
                    pair_id=str(target["pair_id"]),
                    orpha_id_a=parts[0],
                    orpha_id_b=parts[1],
                    target=float(target["literature_similarity"]),
                    paper_count=int(target["unique_papers"]),
                    relationships=str(target["relationships"]),
                )
            )
        result[split_name] = records
    return result


def _clean_set(values: Any) -> set[str]:
    return {
        str(value)
        for value in (values or [])
        if value is not None and str(value).strip()
    }


def _parse_hpo_parents(path: Path) -> Dict[str, set[str]]:
    """Parse canonical HPO ``is_a`` links without adding a new dependency."""

    parents: Dict[str, set[str]] = defaultdict(set)
    current_id = ""
    obsolete = False

    def finish_term() -> None:
        nonlocal current_id, obsolete
        if current_id and obsolete:
            parents.pop(current_id, None)
        current_id = ""
        obsolete = False

    with path.open("r", encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if line == "[Term]":
                finish_term()
            elif line.startswith("[") and line.endswith("]"):
                finish_term()
            elif line.startswith("id: HP:"):
                current_id = line.split("id:", 1)[1].strip()
                parents.setdefault(current_id, set())
            elif current_id and line.startswith("is_a: HP:"):
                parent = line.split("is_a:", 1)[1].split("!", 1)[0].strip()
                parents[current_id].add(parent)
            elif current_id and line == "is_obsolete: true":
                obsolete = True
    finish_term()
    return dict(parents)


def _make_ancestor_lookup(
    parents: Mapping[str, set[str]],
) -> Callable[[str], frozenset[str]]:
    @lru_cache(maxsize=None)
    def ancestors(term: str) -> frozenset[str]:
        found = {term}
        stack = list(parents.get(term, ()))
        while stack:
            parent = stack.pop()
            if parent in found:
                continue
            found.add(parent)
            stack.extend(parents.get(parent, ()))
        return frozenset(found)

    return ancestors


def _load_gene_sets(
    path: Path,
    disease_ids: set[str],
) -> tuple[Dict[str, set[str]], Dict[str, set[str]], Dict[str, int]]:
    """Scan evidence once, retaining curated and high-confidence OT genes."""

    trusted: Dict[str, set[str]] = defaultdict(set)
    high_confidence: Dict[str, set[str]] = defaultdict(set)
    counts = Counter()
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(
        batch_size=131072,
        columns=("orpha_id", "gene_symbol", "source", "association_score"),
    ):
        for row in batch.to_pylist():
            orpha_id = str(row["orpha_id"])
            if orpha_id not in disease_ids:
                continue
            gene = str(row.get("gene_symbol") or "").strip()
            if not gene:
                continue
            source = str(row.get("source") or "")
            if source in TRUSTED_GENE_SOURCES:
                trusted[orpha_id].add(gene)
                counts["trusted_evidence_rows"] += 1
            elif source == OPEN_TARGETS_SOURCE:
                score = row.get("association_score")
                if (
                    score is not None
                    and float(score) >= OPEN_TARGETS_HIGH_CONFIDENCE
                ):
                    high_confidence[orpha_id].add(gene)
                    counts["high_confidence_ot_rows"] += 1
    return dict(trusted), dict(high_confidence), dict(counts)


def build_feature_context(
    feature_dir: Path,
    hpo_obo: Path,
    required_disease_ids: set[str],
) -> tuple[FeatureContext, Dict[str, Any]]:
    """Load source features and construct ontology-aware disease profiles."""

    disease_columns = [
        "orpha_id",
        "name",
        "description",
        "disorder_type",
        "disorder_group",
        "category_ids",
        "body_system_ids",
        "ontology_parent_ids",
        "preferential_parent_ids",
        "hpo_ids",
        "inheritance",
        "onset",
        "prevalence_estimated_per_person",
        "prevalence_class",
        "prevalence_region",
        "approved_drug_ids",
        "drug_ids",
        "treatment_types",
    ]
    disease_rows = pq.read_table(
        feature_dir / "diseases.parquet",
        columns=disease_columns,
    ).to_pylist()
    all_diseases = {
        str(row["orpha_id"]): row
        for row in disease_rows
    }
    missing = sorted(required_disease_ids - set(all_diseases))
    if missing:
        raise KeyError(
            f"{len(missing)} split disease IDs are absent from diseases.parquet; "
            f"first: {missing[:5]}"
        )

    hpo_parents = _parse_hpo_parents(hpo_obo)
    hpo_ancestors = _make_ancestor_lookup(hpo_parents)
    annotation_counts: Counter[str] = Counter()
    annotated_diseases = 0
    for row in disease_rows:
        terms = _clean_set(row.get("hpo_ids"))
        if not terms:
            continue
        annotated_diseases += 1
        expanded: set[str] = set()
        for term in terms:
            expanded.update(hpo_ancestors(term))
        annotation_counts.update(expanded)
    denominator = annotated_diseases + 1.0
    hpo_ic = {
        term: -math.log((count + 1.0) / denominator)
        for term, count in annotation_counts.items()
    }
    hpo_profiles: Dict[str, Dict[str, float]] = {}
    for orpha_id in required_disease_ids:
        profile_terms: set[str] = set()
        for term in _clean_set(all_diseases[orpha_id].get("hpo_ids")):
            profile_terms.update(hpo_ancestors(term))
        hpo_profiles[orpha_id] = {
            term: hpo_ic.get(term, 0.0)
            for term in profile_terms
            if hpo_ic.get(term, 0.0) > 0
        }

    trusted_genes, high_confidence_genes, gene_counts = _load_gene_sets(
        feature_dir / "genes.parquet",
        required_disease_ids,
    )

    descriptions = [str(row.get("description") or "") for row in disease_rows]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=40000,
        sublinear_tf=True,
        norm="l2",
    )
    description_matrix = vectorizer.fit_transform(descriptions).tocsr()
    description_index = {
        str(row["orpha_id"]): index
        for index, row in enumerate(disease_rows)
    }

    context = FeatureContext(
        diseases={
            orpha_id: all_diseases[orpha_id]
            for orpha_id in required_disease_ids
        },
        hpo_ancestors=hpo_ancestors,
        hpo_ic=hpo_ic,
        hpo_profiles=hpo_profiles,
        trusted_genes=trusted_genes,
        high_confidence_ot_genes=high_confidence_genes,
        description_matrix=description_matrix,
        description_index=description_index,
    )
    manifest = {
        "diseases_in_catalog": len(disease_rows),
        "required_diseases": len(required_disease_ids),
        "annotated_hpo_diseases": annotated_diseases,
        "hpo_terms_with_information_content": len(hpo_ic),
        "description_vocabulary_size": len(vectorizer.vocabulary_),
        "description_nonempty_diseases": sum(bool(text.strip()) for text in descriptions),
        "trusted_gene_sources": sorted(TRUSTED_GENE_SOURCES),
        "open_targets_high_confidence_threshold": OPEN_TARGETS_HIGH_CONFIDENCE,
        "trusted_gene_diseases": len(trusted_genes),
        "high_confidence_ot_gene_diseases": len(high_confidence_genes),
        **gene_counts,
    }
    return context, manifest


def _set_statistics(left: set[str], right: set[str]) -> list[float]:
    if not left or not right:
        return [0.0] * len(SET_STAT_SUFFIXES)
    shared_count = len(left & right)
    union_count = len(left | right)
    smaller = min(len(left), len(right))
    larger = max(len(left), len(right))
    return [
        1.0,
        shared_count / union_count,
        shared_count / smaller,
        smaller / larger,
        math.log1p(shared_count),
        math.log1p(union_count),
    ]


def _weighted_profile_statistics(
    left: Mapping[str, float],
    right: Mapping[str, float],
) -> tuple[float, float]:
    if not left or not right:
        return 0.0, 0.0
    shared = set(left) & set(right)
    union = set(left) | set(right)
    numerator = sum(min(left.get(term, 0.0), right.get(term, 0.0)) for term in shared)
    denominator = sum(max(left.get(term, 0.0), right.get(term, 0.0)) for term in union)
    weighted_jaccard = numerator / denominator if denominator else 0.0
    dot = sum(left[term] * right[term] for term in shared)
    norm_left = math.sqrt(sum(value * value for value in left.values()))
    norm_right = math.sqrt(sum(value * value for value in right.values()))
    cosine = dot / (norm_left * norm_right) if norm_left and norm_right else 0.0
    return weighted_jaccard, cosine


def _description_similarity(
    context: FeatureContext,
    orpha_id_a: str,
    orpha_id_b: str,
) -> tuple[float, float]:
    index_a = context.description_index[orpha_id_a]
    index_b = context.description_index[orpha_id_b]
    row_a = context.description_matrix.getrow(index_a)
    row_b = context.description_matrix.getrow(index_b)
    available = bool(row_a.nnz and row_b.nnz)
    cosine = float(row_a.multiply(row_b).sum()) if available else 0.0
    return float(available), cosine


def improved_feature_names() -> list[str]:
    family_names = [
        "phenotypes_exact",
        "phenotypes_ancestors",
        "genes_trusted",
        "genes_high_confidence_ot",
        "genes_curated_union",
        "categories",
        "ontology_parents",
        "preferential_parents",
        "body_systems",
        "inheritance",
        "onset",
        "approved_drugs",
        "all_drugs",
        "treatment_types",
    ]
    return [
        *(
            f"{family}_{suffix}"
            for family in family_names
            for suffix in SET_STAT_SUFFIXES
        ),
        "phenotypes_ic_weighted_jaccard",
        "phenotypes_ic_cosine",
        "prevalence_both_available",
        "prevalence_log10_difference",
        "prevalence_similarity",
        "prevalence_class_match",
        "prevalence_region_match",
        "disorder_type_match",
        "disorder_group_match",
        "description_both_available",
        "description_tfidf_cosine",
    ]


def extract_improved_features(
    context: FeatureContext,
    orpha_id_a: str,
    orpha_id_b: str,
) -> np.ndarray:
    """Build a compact, interpretable, ontology-aware pair vector."""

    left = context.diseases[orpha_id_a]
    right = context.diseases[orpha_id_b]
    left_hpo = _clean_set(left.get("hpo_ids"))
    right_hpo = _clean_set(right.get("hpo_ids"))
    left_ancestors: set[str] = set()
    right_ancestors: set[str] = set()
    for term in left_hpo:
        left_ancestors.update(context.hpo_ancestors(term))
    for term in right_hpo:
        right_ancestors.update(context.hpo_ancestors(term))

    left_trusted = context.trusted_genes.get(orpha_id_a, set())
    right_trusted = context.trusted_genes.get(orpha_id_b, set())
    left_high_confidence = context.high_confidence_ot_genes.get(orpha_id_a, set())
    right_high_confidence = context.high_confidence_ot_genes.get(orpha_id_b, set())
    family_sets = [
        (left_hpo, right_hpo),
        (left_ancestors, right_ancestors),
        (left_trusted, right_trusted),
        (left_high_confidence, right_high_confidence),
        (
            left_trusted | left_high_confidence,
            right_trusted | right_high_confidence,
        ),
        (_clean_set(left.get("category_ids")), _clean_set(right.get("category_ids"))),
        (
            _clean_set(left.get("ontology_parent_ids")),
            _clean_set(right.get("ontology_parent_ids")),
        ),
        (
            _clean_set(left.get("preferential_parent_ids")),
            _clean_set(right.get("preferential_parent_ids")),
        ),
        (
            _clean_set(left.get("body_system_ids")),
            _clean_set(right.get("body_system_ids")),
        ),
        (_clean_set(left.get("inheritance")), _clean_set(right.get("inheritance"))),
        (_clean_set(left.get("onset")), _clean_set(right.get("onset"))),
        (
            _clean_set(left.get("approved_drug_ids")),
            _clean_set(right.get("approved_drug_ids")),
        ),
        (_clean_set(left.get("drug_ids")), _clean_set(right.get("drug_ids"))),
        (
            _clean_set(left.get("treatment_types")),
            _clean_set(right.get("treatment_types")),
        ),
    ]
    values = [
        value
        for left_set, right_set in family_sets
        for value in _set_statistics(left_set, right_set)
    ]

    weighted_jaccard, hpo_cosine = _weighted_profile_statistics(
        context.hpo_profiles.get(orpha_id_a, {}),
        context.hpo_profiles.get(orpha_id_b, {}),
    )
    values.extend([weighted_jaccard, hpo_cosine])

    prevalence_a = left.get("prevalence_estimated_per_person")
    prevalence_b = right.get("prevalence_estimated_per_person")
    prevalence_available = bool(
        prevalence_a is not None
        and prevalence_b is not None
        and float(prevalence_a) > 0
        and float(prevalence_b) > 0
    )
    log_difference = (
        abs(math.log10(float(prevalence_a)) - math.log10(float(prevalence_b)))
        if prevalence_available
        else 0.0
    )
    prevalence_similarity = math.exp(-log_difference) if prevalence_available else 0.0
    left_class = str(left.get("prevalence_class") or "")
    right_class = str(right.get("prevalence_class") or "")
    left_region = str(left.get("prevalence_region") or "")
    right_region = str(right.get("prevalence_region") or "")
    values.extend(
        [
            float(prevalence_available),
            log_difference,
            prevalence_similarity,
            float(bool(left_class and right_class and left_class == right_class)),
            float(bool(left_region and right_region and left_region == right_region)),
            float(
                bool(
                    left.get("disorder_type")
                    and left.get("disorder_type") == right.get("disorder_type")
                )
            ),
            float(
                bool(
                    left.get("disorder_group")
                    and left.get("disorder_group") == right.get("disorder_group")
                )
            ),
        ]
    )
    description_available, description_cosine = _description_similarity(
        context,
        orpha_id_a,
        orpha_id_b,
    )
    values.extend([description_available, description_cosine])

    names = improved_feature_names()
    if len(values) != len(names):
        raise RuntimeError(
            f"Extracted {len(values)} improved values for {len(names)} names."
        )
    return np.asarray(values, dtype=np.float64)


def _model_candidates() -> list[Candidate]:
    candidates = [
        Candidate("ridge", {"alpha": alpha})
        for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
    ]
    candidates.extend(
        Candidate(
            "extra_trees",
            {
                "max_features": max_features,
                "min_samples_leaf": min_samples_leaf,
                "max_depth": max_depth,
            },
        )
        for max_features in ("sqrt", 0.5, 1.0)
        for min_samples_leaf in (2, 5)
        for max_depth in (None, 12)
    )
    candidates.extend(
        Candidate(
            "hist_gradient_boosting",
            {
                "learning_rate": learning_rate,
                "max_leaf_nodes": max_leaf_nodes,
                "l2_regularization": l2_regularization,
            },
        )
        for learning_rate, max_leaf_nodes, l2_regularization in (
            (0.05, 15, 1.0),
            (0.05, 31, 10.0),
            (0.1, 15, 10.0),
            (0.1, 31, 30.0),
        )
    )
    return candidates


def _make_model(candidate: Candidate) -> Any:
    if candidate.name == "ridge":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=float(candidate.parameters["alpha"]))),
            ]
        )
    if candidate.name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=400,
            random_state=20261004,
            n_jobs=1,
            **candidate.parameters,
        )
    if candidate.name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            random_state=20261004,
            max_iter=250,
            min_samples_leaf=10,
            **candidate.parameters,
        )
    raise ValueError(f"Unknown model candidate {candidate.name!r}.")


def _clip_predictions(model: Any, features: np.ndarray) -> np.ndarray:
    return np.clip(
        np.asarray(model.predict(features), dtype=np.float64),
        0.0,
        1.0,
    )


def select_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
) -> tuple[Candidate, Any, list[Dict[str, Any]]]:
    """Select one predefined model by validation RMSE, then MAE."""

    best: Optional[tuple[tuple[float, float, str], Candidate, Any]] = None
    trials: list[Dict[str, Any]] = []
    for candidate in _model_candidates():
        model = _make_model(candidate)
        model.fit(train_x, train_y)
        predictions = _clip_predictions(model, validation_x)
        metrics = regression_metrics(validation_y, predictions)
        row = {
            "model": candidate.name,
            "parameters": candidate.parameters,
            **metrics,
        }
        trials.append(row)
        key = (
            float(metrics["root_mean_squared_error"]),
            float(metrics["mean_absolute_error"]),
            json.dumps(
                {"model": candidate.name, **candidate.parameters},
                sort_keys=True,
            ),
        )
        if best is None or key < best[0]:
            best = (key, candidate, model)
    if best is None:
        raise RuntimeError("No model candidates were evaluated.")
    return best[1], best[2], trials


def _build_feature_matrices(
    records: Mapping[str, Sequence[PairRecord]],
    evaluator: DiseaseDistanceEvaluator,
    context: FeatureContext,
) -> tuple[
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, list[str]],
]:
    improved_names = improved_feature_names()
    text_indices = [
        index
        for index, name in enumerate(improved_names)
        if name.startswith("description_")
    ]
    structured_indices = [
        index
        for index in range(len(improved_names))
        if index not in set(text_indices)
    ]
    names = {
        "current_249": list(CURRENT_FEATURE_NAMES),
        "improved_structured": [improved_names[index] for index in structured_indices],
        "improved_full": improved_names,
        "combined_full": [*CURRENT_FEATURE_NAMES, *improved_names],
    }
    matrices: Dict[str, Dict[str, np.ndarray]] = {
        variant: {}
        for variant in names
    }
    for split_name in SPLIT_NAMES:
        current_rows: list[np.ndarray] = []
        improved_rows: list[np.ndarray] = []
        for record in records[split_name]:
            current, _ = extract_pair_features(
                evaluator,
                record.orpha_id_a,
                record.orpha_id_b,
            )
            improved = extract_improved_features(
                context,
                record.orpha_id_a,
                record.orpha_id_b,
            )
            current_rows.append(current)
            improved_rows.append(improved)
        current_matrix = np.vstack(current_rows)
        improved_matrix = np.vstack(improved_rows)
        matrices["current_249"][split_name] = current_matrix
        matrices["improved_structured"][split_name] = improved_matrix[
            :,
            structured_indices,
        ]
        matrices["improved_full"][split_name] = improved_matrix
        matrices["combined_full"][split_name] = np.hstack(
            [current_matrix, improved_matrix]
        )
    return matrices, names


def benchmark_variants(
    records: Mapping[str, Sequence[PairRecord]],
    matrices: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[Dict[str, Any], Dict[str, Any], list[Dict[str, Any]]]:
    """Tune on validation, refit on train+validation, and test once."""

    targets = {
        split_name: np.asarray(
            [record.target for record in records[split_name]],
            dtype=np.float64,
        )
        for split_name in SPLIT_NAMES
    }
    model_artifacts: Dict[str, Any] = {}
    results: Dict[str, Any] = {}
    prediction_rows: list[Dict[str, Any]] = []

    train_mean = float(np.mean(targets["train"]))
    validation_baseline_predictions = np.full_like(targets["validation"], train_mean)
    final_mean = float(np.mean(np.concatenate([targets["train"], targets["validation"]])))
    test_baseline_predictions = np.full_like(targets["test"], final_mean)
    results["mean_baseline"] = {
        "validation": regression_metrics(
            targets["validation"],
            validation_baseline_predictions,
        ),
        "test": regression_metrics(targets["test"], test_baseline_predictions),
        "final_train_validation_mean": final_mean,
    }

    test_predictions: Dict[str, np.ndarray] = {
        "mean_baseline": test_baseline_predictions,
    }
    for variant, split_matrices in matrices.items():
        selected, _, trials = select_model(
            split_matrices["train"],
            targets["train"],
            split_matrices["validation"],
            targets["validation"],
        )
        validation_best = min(
            trials,
            key=lambda trial: (
                float(trial["root_mean_squared_error"]),
                float(trial["mean_absolute_error"]),
            ),
        )
        final_model = _make_model(selected)
        final_model.fit(
            np.vstack(
                [
                    split_matrices["train"],
                    split_matrices["validation"],
                ]
            ),
            np.concatenate([targets["train"], targets["validation"]]),
        )
        predicted = _clip_predictions(final_model, split_matrices["test"])
        test_predictions[variant] = predicted
        results[variant] = {
            "feature_count": int(split_matrices["train"].shape[1]),
            "selected_model": selected.name,
            "selected_parameters": selected.parameters,
            "validation": {
                key: value
                for key, value in validation_best.items()
                if key not in {"model", "parameters"}
            },
            "test": regression_metrics(targets["test"], predicted),
            "validation_trials": trials,
        }
        model_artifacts[variant] = final_model

    selected_improved = min(
        PRIMARY_IMPROVED_VARIANTS,
        key=lambda variant: float(
            results[variant]["validation"]["root_mean_squared_error"]
        ),
    )
    results["primary_comparison"] = {
        "current_variant": "current_249",
        "selected_improved_variant": selected_improved,
        "selection_rule": (
            "Lower validation RMSE between improved_structured and improved_full; "
            "test metrics were not used for this choice."
        ),
    }

    for index, record in enumerate(records["test"]):
        row: Dict[str, Any] = {
            "split": "test",
            "pair_id": record.pair_id,
            "orpha_id_a": record.orpha_id_a,
            "orpha_id_b": record.orpha_id_b,
            "target": record.target,
            "paper_count": record.paper_count,
            "relationships": record.relationships,
        }
        for variant, predictions in test_predictions.items():
            row[f"prediction_{variant}"] = float(predictions[index])
            row[f"absolute_error_{variant}"] = abs(
                float(predictions[index]) - record.target
            )
        prediction_rows.append(row)

    return results, model_artifacts, prediction_rows


def _paired_bootstrap(
    truth: np.ndarray,
    current_predictions: np.ndarray,
    improved_predictions: np.ndarray,
    *,
    repetitions: int = 20000,
) -> Dict[str, Any]:
    """Bootstrap paired test examples; positive deltas favor the improvement."""

    rng = np.random.default_rng(20261004)
    current_squared = (current_predictions - truth) ** 2
    improved_squared = (improved_predictions - truth) ** 2
    current_absolute = np.abs(current_predictions - truth)
    improved_absolute = np.abs(improved_predictions - truth)
    rmse_deltas = np.empty(repetitions, dtype=np.float64)
    mae_deltas = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        indices = rng.integers(0, len(truth), size=len(truth))
        rmse_deltas[repetition] = math.sqrt(
            float(np.mean(current_squared[indices]))
        ) - math.sqrt(float(np.mean(improved_squared[indices])))
        mae_deltas[repetition] = float(
            np.mean(current_absolute[indices] - improved_absolute[indices])
        )
    observed_rmse_delta = math.sqrt(float(np.mean(current_squared))) - math.sqrt(
        float(np.mean(improved_squared))
    )
    observed_mae_delta = float(np.mean(current_absolute - improved_absolute))
    return {
        "interpretation": "Positive deltas favor the selected improved variant.",
        "repetitions": repetitions,
        "test_examples": len(truth),
        "rmse_delta": observed_rmse_delta,
        "rmse_delta_95_percentile_interval": np.quantile(
            rmse_deltas,
            [0.025, 0.975],
        ).tolist(),
        "rmse_probability_improved": float(np.mean(rmse_deltas > 0)),
        "mae_delta": observed_mae_delta,
        "mae_delta_95_percentile_interval": np.quantile(
            mae_deltas,
            [0.025, 0.975],
        ).tolist(),
        "mae_probability_improved": float(np.mean(mae_deltas > 0)),
    }


def _matched_control_ids(
    record: PairRecord,
    context: FeatureContext,
    candidate_ids: Sequence[str],
    known_pairs: set[frozenset[str]],
    *,
    count: int,
) -> list[str]:
    """Choose deterministic, broad-system-matched unobserved candidates."""

    anchor = record.orpha_id_a
    positive = record.orpha_id_b
    anchor_row = context.diseases[anchor]
    anchor_systems = _clean_set(anchor_row.get("body_system_ids"))
    anchor_hpo_count = len(_clean_set(anchor_row.get("hpo_ids")))
    ranked: list[tuple[tuple[float, ...], str]] = []
    for candidate in candidate_ids:
        if candidate in {anchor, positive}:
            continue
        pair = frozenset((anchor, candidate))
        if pair in known_pairs:
            continue
        candidate_row = context.diseases[candidate]
        candidate_systems = _clean_set(candidate_row.get("body_system_ids"))
        shared_systems = len(anchor_systems & candidate_systems)
        if anchor_systems and candidate_systems and shared_systems == 0:
            continue
        union_systems = len(anchor_systems | candidate_systems)
        body_jaccard = shared_systems / union_systems if union_systems else 0.0
        same_group = float(
            bool(
                anchor_row.get("disorder_group")
                and anchor_row.get("disorder_group")
                == candidate_row.get("disorder_group")
            )
        )
        candidate_hpo_count = len(_clean_set(candidate_row.get("hpo_ids")))
        annotation_gap = abs(
            math.log1p(anchor_hpo_count) - math.log1p(candidate_hpo_count)
        )
        tie_breaker = int(
            hashlib.blake2b(
                f"{record.pair_id}\0{candidate}".encode("utf-8"),
                digest_size=8,
            ).hexdigest(),
            16,
        )
        key = (
            -body_jaccard,
            -same_group,
            annotation_gap,
            float(tie_breaker),
        )
        ranked.append((key, candidate))
    ranked.sort()
    return [candidate for _, candidate in ranked[:count]]


def _average_descending_rank(scores: np.ndarray, true_index: int) -> float:
    true_score = scores[true_index]
    greater = int(np.sum(scores > true_score))
    equal_other = int(np.sum(scores == true_score)) - 1
    return 1.0 + greater + 0.5 * equal_other


def _ranking_metrics(ranks: Sequence[float]) -> Dict[str, Any]:
    values = np.asarray(ranks, dtype=np.float64)
    return {
        "queries": int(len(values)),
        "mean_reciprocal_rank": float(np.mean(1.0 / values)),
        "mean_rank": float(np.mean(values)),
        "median_rank": float(np.median(values)),
        "hits_at_1": float(np.mean(values <= 1.0)),
        "hits_at_5": float(np.mean(values <= 5.0)),
    }


def _paired_ranking_bootstrap(
    current_ranks: Sequence[float],
    improved_ranks: Sequence[float],
    *,
    repetitions: int = 20000,
) -> Dict[str, Any]:
    """Bootstrap query-level MRR deltas; positive values favor improvement."""

    current = np.asarray(current_ranks, dtype=np.float64)
    improved = np.asarray(improved_ranks, dtype=np.float64)
    if len(current) != len(improved) or len(current) == 0:
        raise ValueError("Ranking bootstrap requires equal non-empty rank arrays.")
    reciprocal_deltas = (1.0 / improved) - (1.0 / current)
    rng = np.random.default_rng(20261004)
    bootstrap_deltas = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        indices = rng.integers(0, len(current), size=len(current))
        bootstrap_deltas[repetition] = float(
            np.mean(reciprocal_deltas[indices])
        )
    return {
        "interpretation": "Positive MRR deltas favor the selected improved model.",
        "repetitions": repetitions,
        "queries": int(len(current)),
        "mrr_delta": float(np.mean(reciprocal_deltas)),
        "mrr_delta_95_percentile_interval": np.quantile(
            bootstrap_deltas,
            [0.025, 0.975],
        ).tolist(),
        "mrr_probability_improved": float(np.mean(bootstrap_deltas > 0)),
    }


def run_matched_control_benchmark(
    records: Mapping[str, Sequence[PairRecord]],
    evaluator: DiseaseDistanceEvaluator,
    context: FeatureContext,
    feature_names: Mapping[str, Sequence[str]],
    model_artifacts: Mapping[str, Any],
    selected_improved_variant: str,
    *,
    controls_per_query: int,
) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    """Rank known high-similarity test edges among matched unobserved controls."""

    all_records = [
        record
        for split_records in records.values()
        for record in split_records
    ]
    known_pairs = {
        frozenset((record.orpha_id_a, record.orpha_id_b))
        for record in all_records
    }
    candidate_ids = sorted(context.diseases)
    test_positives = [
        record
        for record in records["test"]
        if record.target >= 0.8
    ]
    row_outputs: list[Dict[str, Any]] = []
    ranks_by_method: Dict[str, list[float]] = defaultdict(list)

    improved_all_names = improved_feature_names()
    selected_improved_names = list(feature_names[selected_improved_variant])
    selected_improved_indices = [
        improved_all_names.index(name)
        for name in selected_improved_names
    ]
    for query_index, record in enumerate(test_positives, start=1):
        controls = _matched_control_ids(
            record,
            context,
            candidate_ids,
            known_pairs,
            count=controls_per_query,
        )
        if len(controls) < controls_per_query:
            continue
        partner_ids = [record.orpha_id_b, *controls]
        current_rows: list[np.ndarray] = []
        improved_rows: list[np.ndarray] = []
        heuristic_scores: list[float] = []
        for partner_id in partner_ids:
            current, calculation = extract_pair_features(
                evaluator,
                record.orpha_id_a,
                partner_id,
            )
            improved_full = extract_improved_features(
                context,
                record.orpha_id_a,
                partner_id,
            )
            current_rows.append(current)
            improved_rows.append(improved_full[selected_improved_indices])
            raw_heuristic = calculation.get("similarity")
            heuristic_scores.append(
                float(raw_heuristic) if raw_heuristic is not None else 0.0
            )
        current_matrix = np.vstack(current_rows)
        improved_matrix = np.vstack(improved_rows)
        method_scores = {
            "current_heuristic": np.asarray(heuristic_scores, dtype=np.float64),
            "current_249_model": _clip_predictions(
                model_artifacts["current_249"],
                current_matrix,
            ),
            "selected_improved_model": _clip_predictions(
                model_artifacts[selected_improved_variant],
                improved_matrix,
            ),
        }
        ranks = {
            method: _average_descending_rank(scores, 0)
            for method, scores in method_scores.items()
        }
        for method, rank in ranks.items():
            ranks_by_method[method].append(rank)
        for partner_index, partner_id in enumerate(partner_ids):
            row_outputs.append(
                {
                    "query_index": query_index,
                    "positive_pair_id": record.pair_id,
                    "anchor_orpha_id": record.orpha_id_a,
                    "candidate_orpha_id": partner_id,
                    "is_literature_positive": int(partner_index == 0),
                    "literature_target": (
                        record.target if partner_index == 0 else ""
                    ),
                    "current_heuristic_score": float(
                        method_scores["current_heuristic"][partner_index]
                    ),
                    "current_249_model_score": float(
                        method_scores["current_249_model"][partner_index]
                    ),
                    "selected_improved_model_score": float(
                        method_scores["selected_improved_model"][partner_index]
                    ),
                    "current_heuristic_positive_rank": ranks["current_heuristic"],
                    "current_249_model_positive_rank": ranks["current_249_model"],
                    "selected_improved_model_positive_rank": ranks[
                        "selected_improved_model"
                    ],
                }
            )

    summary = {
        "controls_per_query": controls_per_query,
        "positive_definition": "test literature target >= 0.8",
        "control_definition": (
            "Same broad body-system when available, annotation-count matched, "
            "and absent from all literature pairs. Controls are unobserved, "
            "not clinically verified negatives."
        ),
        "methods": {
            method: _ranking_metrics(ranks)
            for method, ranks in ranks_by_method.items()
        },
        "paired_bootstrap": _paired_ranking_bootstrap(
            ranks_by_method["current_249_model"],
            ranks_by_method["selected_improved_model"],
        ),
    }
    return summary, row_outputs


def _bar(label: str, value: float, maximum: float, *, lower_is_better: bool) -> str:
    width = 0.0 if maximum <= 0 else min(100.0, max(0.0, value / maximum * 100.0))
    direction = "lower is better" if lower_is_better else "higher is better"
    return (
        '<div class="bar-row">'
        f'<div class="bar-label">{html.escape(label)}</div>'
        '<div class="track"><div class="fill" '
        f'style="width:{width:.1f}%"></div></div>'
        f'<div class="bar-value">{value:.3f}</div>'
        f'<div class="sr-only">{html.escape(direction)}</div>'
        "</div>"
    )


def write_html_report(path: Path, metrics: Mapping[str, Any]) -> None:
    """Write a self-contained visual report without external assets."""

    variants = metrics["global_regression"]["variants"]
    comparison = variants["primary_comparison"]
    selected = comparison["selected_improved_variant"]
    current_test = variants["current_249"]["test"]
    improved_test = variants[selected]["test"]
    baseline_test = variants["mean_baseline"]["test"]
    bootstrap = metrics["global_regression"]["paired_bootstrap"]
    ranking = metrics["matched_control_ranking"]["methods"]
    ranking_bootstrap = metrics["matched_control_ranking"]["paired_bootstrap"]
    regression_bars = "".join(
        [
            _bar(
                "Mean baseline",
                float(baseline_test["root_mean_squared_error"]),
                0.25,
                lower_is_better=True,
            ),
            _bar(
                "Current 249 features",
                float(current_test["root_mean_squared_error"]),
                0.25,
                lower_is_better=True,
            ),
            _bar(
                selected.replace("_", " ").title(),
                float(improved_test["root_mean_squared_error"]),
                0.25,
                lower_is_better=True,
            ),
        ]
    )
    ranking_bars = "".join(
        _bar(
            method.replace("_", " ").title(),
            float(values["mean_reciprocal_rank"]),
            1.0,
            lower_is_better=False,
        )
        for method, values in ranking.items()
    )
    rmse_delta = float(bootstrap["rmse_delta"])
    interval = bootstrap["rmse_delta_95_percentile_interval"]
    conclusion = (
        "Improved"
        if rmse_delta > 0
        else "Did not improve"
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Semantic similarity experiment</title>
<style>
:root {{ color-scheme: light dark; font-family: Inter, system-ui, sans-serif; }}
body {{ margin: 0; background: Canvas; color: CanvasText; }}
main {{ max-width: 1040px; margin: 0 auto; padding: 40px 28px 64px; }}
h1 {{ font-size: 28px; margin: 0 0 8px; }}
h2 {{ font-size: 18px; margin: 32px 0 12px; }}
p {{ line-height: 1.55; max-width: 780px; }}
.lede {{ color: GrayText; margin-top: 0; }}
.verdict {{ display: grid; grid-template-columns: 1fr auto; gap: 20px;
  align-items: end; border-top: 1px solid GrayText; border-bottom: 1px solid GrayText;
  padding: 22px 0; margin: 28px 0; }}
.verdict strong {{ font-size: 24px; }}
.metric {{ font-variant-numeric: tabular-nums; text-align: right; }}
.bar-row {{ display: grid; grid-template-columns: 190px 1fr 64px; gap: 12px;
  align-items: center; margin: 12px 0; }}
.track {{ height: 18px; background: color-mix(in srgb, CanvasText 10%, Canvas); }}
.fill {{ height: 100%; background: AccentColor; }}
.bar-value {{ font-variant-numeric: tabular-nums; text-align: right; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(220px,1fr));
  gap: 18px; margin-top: 18px; }}
.stat {{ padding: 16px 0; border-top: 1px solid color-mix(in srgb, CanvasText 24%, Canvas); }}
.stat b {{ display: block; font-size: 22px; margin-top: 4px; }}
.note {{ color: GrayText; font-size: 13px; }}
.sr-only {{ position: absolute; width: 1px; height: 1px; overflow: hidden;
  clip: rect(0,0,0,0); }}
code {{ font-family: ui-monospace, monospace; }}
@media (max-width: 640px) {{
  .bar-row {{ grid-template-columns: 1fr 54px; }}
  .bar-label {{ grid-column: 1 / -1; }}
  .verdict {{ grid-template-columns: 1fr; }}
  .metric {{ text-align: left; }}
}}
</style>
</head>
<body><main>
<h1>Semantic disease-similarity experiment</h1>
<p class="lede">Current split · 486 train, 62 validation, 59 untouched test pairs</p>
<section class="verdict">
  <div><span>Primary held-out result</span><br>
  <strong>{conclusion} test RMSE</strong></div>
  <div class="metric">Δ RMSE (current − improved)<br><strong>{rmse_delta:+.4f}</strong></div>
</section>
<p>The improved representation was selected between structured-only and
structured-plus-description variants using validation RMSE. The test set was
then evaluated once. Positive deltas favor the improved representation.</p>
<h2>Held-out regression RMSE</h2>
{regression_bars}
<p class="note">Target: paper-averaged literature similarity in [0, 1].
Lower RMSE is better. Source: current deterministic pair split.</p>
<div class="grid">
  <div class="stat">Selected representation<b>{html.escape(selected)}</b></div>
  <div class="stat">Current test R²<b>{float(current_test["r2"]):.3f}</b></div>
  <div class="stat">Improved test R²<b>{float(improved_test["r2"]):.3f}</b></div>
  <div class="stat">Bootstrap P(Δ RMSE &gt; 0)<b>{float(bootstrap["rmse_probability_improved"]):.1%}</b></div>
</div>
<p class="note">Paired bootstrap 95% interval for Δ RMSE:
[{float(interval[0]):+.4f}, {float(interval[1]):+.4f}] across
{int(bootstrap["repetitions"]):,} resamples.</p>
<h2>Matched-control retrieval</h2>
{ranking_bars}
<p class="note">Mean reciprocal rank; higher is better. Each known test edge
with target ≥ 0.8 is ranked against {int(metrics["matched_control_ranking"]["controls_per_query"])}
body-system-matched unobserved candidates. These candidates are not verified
clinical negatives. Paired bootstrap P(Δ MRR &gt; 0):
{float(ranking_bootstrap["mrr_probability_improved"]):.1%}; 95% interval:
[{float(ranking_bootstrap["mrr_delta_95_percentile_interval"][0]):+.3f},
{float(ranking_bootstrap["mrr_delta_95_percentile_interval"][1]):+.3f}].</p>
<h2>What changed</h2>
<p>Exact HPO overlap was augmented with HPO ancestor expansion and
information-content weighting. Broad low-score Open Targets genes were replaced
by curated sources plus associations scoring at least
{OPEN_TARGETS_HIGH_CONFIDENCE:.1f}. The compact representation also adds
ontology parents, all treatments, disease type, and full TF-IDF cosine over
Orphanet descriptions without lossy SVD compression.</p>
<h2>Interpretation boundary</h2>
<p>This experiment can test whether better representations recover the existing
literature labels and rank known edges. It cannot repair subjective abstract-only
annotations, create expert labels, or prove that an unobserved control is
biologically unrelated.</p>
</main></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def run_experiment(
    *,
    feature_dir: Path,
    split_dir: Path,
    hpo_obo: Path,
    output_dir: Path,
    controls_per_query: int,
) -> Dict[str, Any]:
    records = load_records(split_dir)
    required_ids = {
        orpha_id
        for split_records in records.values()
        for record in split_records
        for orpha_id in (record.orpha_id_a, record.orpha_id_b)
    }
    evaluator = DiseaseDistanceEvaluator(feature_dir=feature_dir)
    context, source_manifest = build_feature_context(
        feature_dir,
        hpo_obo,
        required_ids,
    )
    matrices, feature_names = _build_feature_matrices(records, evaluator, context)
    variant_results, model_artifacts, prediction_rows = benchmark_variants(
        records,
        matrices,
    )
    selected_improved = variant_results["primary_comparison"][
        "selected_improved_variant"
    ]
    test_truth = np.asarray(
        [record.target for record in records["test"]],
        dtype=np.float64,
    )
    prediction_by_pair = {
        row["pair_id"]: row
        for row in prediction_rows
    }
    current_predictions = np.asarray(
        [
            prediction_by_pair[record.pair_id]["prediction_current_249"]
            for record in records["test"]
        ],
        dtype=np.float64,
    )
    improved_predictions = np.asarray(
        [
            prediction_by_pair[record.pair_id][f"prediction_{selected_improved}"]
            for record in records["test"]
        ],
        dtype=np.float64,
    )
    bootstrap = _paired_bootstrap(
        test_truth,
        current_predictions,
        improved_predictions,
    )
    ranking_metrics, ranking_rows = run_matched_control_benchmark(
        records,
        evaluator,
        context,
        feature_names,
        model_artifacts,
        selected_improved,
        controls_per_query=controls_per_query,
    )

    split_manifest_path = split_dir / "split_manifest.json"
    metrics: Dict[str, Any] = {
        "schema_version": "1.0.0",
        "experiment": "semantic_similarity_v1",
        "isolation": (
            "All generated artifacts are below the root experiments directory."
        ),
        "inputs": {
            "feature_directory": str(feature_dir),
            "split_directory": str(split_dir),
            "hpo_obo": str(hpo_obo),
            "split_manifest_sha256": _sha256(split_manifest_path),
            "pair_counts": {
                split_name: len(records[split_name])
                for split_name in SPLIT_NAMES
            },
        },
        "source_feature_manifest": source_manifest,
        "feature_variants": {
            variant: {
                "count": len(names),
                "names": list(names),
            }
            for variant, names in feature_names.items()
        },
        "global_regression": {
            "target": "paper-averaged literature similarity",
            "variants": variant_results,
            "paired_bootstrap": bootstrap,
        },
        "matched_control_ranking": ranking_metrics,
        "limitations": [
            "Existing literature scores are retained for direct comparability.",
            "Most source labels were derived from abstracts rather than expert full-text review.",
            "Unobserved matched controls are ranking decoys, not verified unrelated pairs.",
            "The test split has only 59 pairs, so paired uncertainty is reported.",
            "TF-IDF and HPO information content are unsupervised and fit over the full disease catalog.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(
        output_dir / "feature_manifest.json",
        {
            "schema_version": "1.0.0",
            "source_feature_manifest": source_manifest,
            "variants": metrics["feature_variants"],
        },
    )
    prediction_fields = list(prediction_rows[0])
    _write_csv(
        output_dir / "test_predictions.csv",
        prediction_fields,
        prediction_rows,
    )
    if ranking_rows:
        _write_csv(
            output_dir / "matched_control_rankings.csv",
            list(ranking_rows[0]),
            ranking_rows,
        )
    importance_rows: list[Dict[str, Any]] = []
    for variant, model in model_artifacts.items():
        raw_importances = getattr(model, "feature_importances_", None)
        if raw_importances is None:
            continue
        for feature, importance in sorted(
            zip(feature_names[variant], raw_importances),
            key=lambda item: float(item[1]),
            reverse=True,
        ):
            importance_rows.append(
                {
                    "variant": variant,
                    "feature": feature,
                    "impurity_importance": float(importance),
                }
            )
    if importance_rows:
        _write_csv(
            output_dir / "feature_importances.csv",
            ("variant", "feature", "impurity_importance"),
            importance_rows,
        )
    joblib.dump(
        {
            "schema_version": "1.0.0",
            "models": model_artifacts,
            "feature_names": feature_names,
            "selected_improved_variant": selected_improved,
            "note": (
                "Feature context is intentionally not embedded because the HPO, "
                "TF-IDF, and curated-gene transforms are reproducibly rebuilt by "
                "the experiment script."
            ),
        },
        output_dir / "models.joblib",
    )
    write_html_report(output_dir / "report.html", metrics)
    return metrics


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--hpo-obo", type=Path, default=DEFAULT_HPO_OBO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--controls-per-query", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        output_dir = _assert_output_is_isolated(args.output_dir)
        if args.controls_per_query < 1:
            raise ValueError("--controls-per-query must be at least 1.")
        metrics = run_experiment(
            feature_dir=args.feature_dir.expanduser().resolve(),
            split_dir=args.split_dir.expanduser().resolve(),
            hpo_obo=args.hpo_obo.expanduser().resolve(),
            output_dir=output_dir,
            controls_per_query=args.controls_per_query,
        )
    except (FileNotFoundError, KeyError, RuntimeError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    variants = metrics["global_regression"]["variants"]
    selected = variants["primary_comparison"]["selected_improved_variant"]
    current_rmse = variants["current_249"]["test"]["root_mean_squared_error"]
    improved_rmse = variants[selected]["test"]["root_mean_squared_error"]
    bootstrap = metrics["global_regression"]["paired_bootstrap"]
    print(f"Artifacts: {output_dir}")
    print(f"Selected improved variant: {selected}")
    print(f"Current test RMSE: {current_rmse:.4f}")
    print(f"Improved test RMSE: {improved_rmse:.4f}")
    print(
        "Paired delta (current - improved): "
        f"{bootstrap['rmse_delta']:+.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
