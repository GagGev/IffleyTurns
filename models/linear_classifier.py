"""Train an interpretable linear classifier for rare-disease similarity.

Literature observations are aggregated by unordered ORPHA pair so each model
example has one ground-truth score.  A pair is labelled similar when its mean
literature score meets ``--positive-threshold`` (0.5 by default).

Inputs are pairwise similarities, containment, set-size balance, and
availability values derived from the feature families in ``evaluation.py``.
Fixed-width hashed channels also retain which terms are shared or differ
without requiring a vocabulary-sized model matrix.  Logistic-regression
regularization is selected on the validation split; the selected model is then
refit on train+validation and evaluated once on the test split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_WEIGHTS,
    DiseaseDistanceEvaluator,
    SET_FEATURES,
)
from models.evaluation import evaluate_classifier  # noqa: E402


DEFAULT_SPLIT_DIR = PROJECT_ROOT / ".data" / "splits"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".data" / "models" / "linear_classifier"
DEFAULT_C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
FEATURE_FAMILIES = tuple(DEFAULT_WEIGHTS)
SET_FEATURE_FAMILIES = tuple(
    family for family in FEATURE_FAMILIES if family != "prevalence"
)
DENSE_FEATURE_NAMES = (
    *(
        feature_name
        for family in SET_FEATURE_FAMILIES
        for feature_name in (
            f"{family}_similarity",
            f"{family}_available",
            f"{family}_overlap_coefficient",
            f"{family}_size_balance",
        )
    ),
    "prevalence_similarity",
    "prevalence_available",
    "prevalence_log10_difference",
    "prevalence_class_match",
    "prevalence_region_match",
)
HASH_BUCKETS = {
    "phenotypes": {"shared": 32, "different": 24},
    "genes": {"shared": 32, "different": 24},
    "classifications": {"shared": 24, "different": 12},
    "body_systems": {"shared": 16, "different": 8},
    "inheritance": {"shared": 8, "different": 4},
    "onset": {"shared": 8, "different": 4},
    "approved_drugs": {"shared": 16, "different": 4},
}
HASHED_FEATURE_NAMES = tuple(
    f"{family}_{channel}_hash_{bucket:02d}"
    for family in SET_FEATURE_FAMILIES
    for channel in ("shared", "different")
    for bucket in range(HASH_BUCKETS[family][channel])
)
FEATURE_NAMES = (*DENSE_FEATURE_NAMES, *HASHED_FEATURE_NAMES)
MODEL_FILENAME = "model.joblib"
METRICS_FILENAME = "metrics.json"
COEFFICIENTS_FILENAME = "coefficients.csv"
PREDICTIONS_FILENAME = "predictions.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_pair_id(pair_id: str) -> tuple[str, str]:
    parts = pair_id.split("|")
    if len(parts) != 2 or not all(part.startswith("ORPHA:") for part in parts):
        raise ValueError(f"Invalid _pair_id {pair_id!r}")
    return parts[0], parts[1]


def load_pair_targets(path: Path, expected_split: str) -> list[Dict[str, Any]]:
    """Aggregate paper-level scores into one target per ORPHA pair."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Split file not found: {path}. Run splits.py first."
        )

    grouped: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"_pair_id", "_split", "similarity_score", "relationship"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path} is missing required column(s): "
                + ", ".join(sorted(missing))
            )

        for row_number, row in enumerate(reader, start=2):
            if row["_split"] != expected_split:
                raise ValueError(
                    f"{path}:{row_number} declares split {row['_split']!r}; "
                    f"expected {expected_split!r}."
                )
            pair_id = row["_pair_id"].strip()
            _parse_pair_id(pair_id)
            try:
                score = float(row["similarity_score"])
            except ValueError as error:
                raise ValueError(
                    f"{path}:{row_number} has an invalid similarity_score."
                ) from error
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"{path}:{row_number} similarity_score must be in [0, 1]."
                )

            group = grouped.setdefault(
                pair_id,
                {
                    "pair_id": pair_id,
                    "paper_scores": {},
                    "relationships": set(),
                    "source_rows": 0,
                },
            )
            paper_id = row.get("_paper_id", "").strip() or f"ROW:{row_number}"
            group["paper_scores"].setdefault(paper_id, []).append(score)
            group["source_rows"] += 1
            if row["relationship"].strip():
                group["relationships"].add(row["relationship"].strip())

    targets: list[Dict[str, Any]] = []
    for pair_id, group in sorted(grouped.items()):
        # A paper gets one vote even if the source CSV contains multiple rows
        # for that paper/pair (for example, one row per evidence dimension).
        paper_scores = [
            statistics.fmean(scores)
            for scores in group["paper_scores"].values()
        ]
        targets.append(
            {
                "pair_id": pair_id,
                "literature_similarity": statistics.fmean(paper_scores),
                "literature_similarity_min": min(paper_scores),
                "literature_similarity_max": max(paper_scores),
                "literature_observations": len(paper_scores),
                "unique_papers": len(paper_scores),
                "source_rows": group["source_rows"],
                "relationships": " | ".join(sorted(group["relationships"])),
            }
        )
    return targets


def _hashed_terms(
    terms: set[str],
    *,
    buckets: int,
    namespace: str,
) -> list[float]:
    """Project a term set into deterministic signed, length-normalized bins."""

    values = [0.0] * buckets
    if not terms:
        return values

    scale = 1.0 / math.sqrt(len(terms))
    for term in sorted(terms):
        digest = hashlib.blake2b(
            f"{namespace}\0{term}".encode("utf-8"),
            digest_size=8,
        ).digest()
        bucket = int.from_bytes(digest[:4], "big") % buckets
        sign = 1.0 if digest[4] & 1 else -1.0
        values[bucket] += sign * scale
    return values


def extract_pair_features(
    evaluator: DiseaseDistanceEvaluator,
    disease_a_id: str,
    disease_b_id: str,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    """Extract fixed-width model features for one ORPHA pair."""

    result = evaluator.calculate_distance(
        disease_a_id,
        disease_b_id,
        allow_no_comparable=True,
    )
    components = {
        component["feature"]: component for component in result["components"]
    }

    values: list[float] = []
    for family in SET_FEATURE_FAMILIES:
        component = components[family]
        available = bool(component["available"])
        left_count = int(component.get("left_count", 0))
        right_count = int(component.get("right_count", 0))
        shared_count = int(component.get("shared_count", 0))
        smaller_count = min(left_count, right_count)
        larger_count = max(left_count, right_count)
        values.extend(
            [
                float(component["similarity"]) if available else 0.0,
                1.0 if available else 0.0,
                (
                    shared_count / smaller_count
                    if available and smaller_count
                    else 0.0
                ),
                (
                    smaller_count / larger_count
                    if available and larger_count
                    else 0.0
                ),
            ]
        )

    prevalence = components["prevalence"]
    prevalence_available = bool(prevalence["available"])
    left_class = str(prevalence.get("left_class") or "")
    right_class = str(prevalence.get("right_class") or "")
    left_region = str(prevalence.get("left_region") or "")
    right_region = str(prevalence.get("right_region") or "")
    values.extend(
        [
            float(prevalence["similarity"]) if prevalence_available else 0.0,
            1.0 if prevalence_available else 0.0,
            (
                float(prevalence["orders_of_magnitude_difference"])
                if prevalence_available
                else 0.0
            ),
            float(bool(left_class and right_class and left_class == right_class)),
            float(
                bool(left_region and right_region and left_region == right_region)
            ),
        ]
    )

    disease_a = evaluator.get_disease(disease_a_id)
    disease_b = evaluator.get_disease(disease_b_id)
    for family in SET_FEATURE_FAMILIES:
        column = SET_FEATURES[family]
        terms_a = {
            str(value)
            for value in disease_a.get(column) or []
            if value not in (None, "")
        }
        terms_b = {
            str(value)
            for value in disease_b.get(column) or []
            if value not in (None, "")
        }
        values.extend(
            _hashed_terms(
                terms_a & terms_b,
                buckets=HASH_BUCKETS[family]["shared"],
                namespace=f"{family}:shared",
            )
        )
        values.extend(
            _hashed_terms(
                terms_a ^ terms_b,
                buckets=HASH_BUCKETS[family]["different"],
                namespace=f"{family}:different",
            )
        )
    if len(values) != len(FEATURE_NAMES):
        raise RuntimeError(
            f"Extracted {len(values)} values for {len(FEATURE_NAMES)} feature names."
        )
    return np.asarray(values, dtype=np.float64), result


def build_dataset(
    targets: Sequence[Mapping[str, Any]],
    evaluator: DiseaseDistanceEvaluator,
    *,
    positive_threshold: float,
) -> tuple[np.ndarray, np.ndarray, list[Dict[str, Any]]]:
    """Join pair targets to generated pairwise feature vectors."""

    feature_rows: list[np.ndarray] = []
    labels: list[int] = []
    metadata: list[Dict[str, Any]] = []

    for target in targets:
        disease_a_id, disease_b_id = _parse_pair_id(str(target["pair_id"]))
        features, result = extract_pair_features(
            evaluator,
            disease_a_id,
            disease_b_id,
        )
        literature_similarity = float(target["literature_similarity"])
        label = int(literature_similarity >= positive_threshold)
        feature_rows.append(features)
        labels.append(label)
        metadata.append(
            {
                **target,
                "orpha_id_a": disease_a_id,
                "orpha_id_b": disease_b_id,
                "disease_a": result["disease_a"]["name"],
                "disease_b": result["disease_b"]["name"],
                "label": label,
                "heuristic_similarity": (
                    float(result["similarity"])
                    if result["similarity"] is not None
                    else ""
                ),
            }
        )

    if not feature_rows:
        raise ValueError("A split contains no usable ORPHA pairs.")
    return (
        np.vstack(feature_rows),
        np.asarray(labels, dtype=np.int64),
        metadata,
    )


def _validate_binary_training_labels(labels: np.ndarray) -> None:
    classes = set(int(value) for value in labels)
    if classes != {0, 1}:
        raise ValueError(
            "Training data must contain both classes after pair aggregation; "
            f"found {sorted(classes)}."
        )


def make_pipeline(c_value: float, random_state: int) -> Pipeline:
    """Create the standardized L2 logistic-regression pipeline."""

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=random_state,
                    solver="liblinear",
                ),
            ),
        ]
    )


def select_regularization(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    *,
    c_grid: Sequence[float],
    random_state: int,
) -> tuple[float, Pipeline, list[Dict[str, Any]]]:
    """Select C using validation ROC AUC, then balanced accuracy."""

    trials: list[Dict[str, Any]] = []
    best_key: Optional[tuple[float, float, float]] = None
    best_c = 0.0
    best_model: Optional[Pipeline] = None

    for c_value in c_grid:
        model = make_pipeline(c_value, random_state)
        model.fit(train_features, train_labels)
        evaluation = evaluate_classifier(
            model,
            validation_features,
            validation_labels,
            probability_threshold=0.5,
        )
        metrics = evaluation.metrics
        trials.append({"c": c_value, **metrics})
        roc_auc = metrics["roc_auc"]
        # Prefer simpler (smaller-C) models when validation metrics tie.
        selection_key = (
            float(roc_auc) if roc_auc is not None else -1.0,
            float(metrics["balanced_accuracy"]),
            -c_value,
        )
        if best_key is None or selection_key > best_key:
            best_key = selection_key
            best_c = c_value
            best_model = model

    if best_model is None:
        raise ValueError("The regularization grid is empty.")
    return best_c, best_model, trials


def _prediction_rows(
    split_name: str,
    metadata: Sequence[Mapping[str, Any]],
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for item, probability, prediction in zip(
        metadata,
        probabilities,
        predictions,
    ):
        rows.append(
            {
                "split": split_name,
                "pair_id": item["pair_id"],
                "orpha_id_a": item["orpha_id_a"],
                "orpha_id_b": item["orpha_id_b"],
                "disease_a": item["disease_a"],
                "disease_b": item["disease_b"],
                "literature_similarity": item["literature_similarity"],
                "literature_similarity_min": item["literature_similarity_min"],
                "literature_similarity_max": item["literature_similarity_max"],
                "literature_observations": item["literature_observations"],
                "unique_papers": item["unique_papers"],
                "source_rows": item["source_rows"],
                "relationships": item["relationships"],
                "true_label": item["label"],
                "predicted_label": int(prediction),
                "probability_similar": float(probability),
                "heuristic_similarity": item["heuristic_similarity"],
            }
        )
    return rows


def _coefficient_rows(model: Pipeline) -> list[Dict[str, Any]]:
    scaler: StandardScaler = model.named_steps["scaler"]
    classifier: LogisticRegression = model.named_steps["classifier"]
    standardized = classifier.coef_[0]
    raw_unit = standardized / scaler.scale_
    rows = [
        {
            "feature": feature_name,
            "standardized_coefficient": float(standardized_coefficient),
            "raw_unit_coefficient": float(raw_coefficient),
            "odds_ratio_per_raw_unit": float(
                math.exp(float(np.clip(raw_coefficient, -50.0, 50.0)))
            ),
        }
        for feature_name, standardized_coefficient, raw_coefficient in zip(
            FEATURE_NAMES,
            standardized,
            raw_unit,
        )
    ]
    return sorted(
        rows,
        key=lambda row: abs(row["standardized_coefficient"]),
        reverse=True,
    )


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(value, file, indent=2, sort_keys=True)
            file.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_csv_atomic(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _dump_model_atomic(path: Path, artifact: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        joblib.dump(artifact, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _positive_threshold(value: str) -> float:
    try:
        threshold = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("threshold must be a number") from error
    if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise argparse.ArgumentTypeError("threshold must be strictly between 0 and 1")
    return threshold


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be a number") from error
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return number


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=DEFAULT_SPLIT_DIR,
        help=f"directory produced by splits.py (default: {DEFAULT_SPLIT_DIR})",
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=DEFAULT_FEATURE_DIR,
        help=f"generated feature directory (default: {DEFAULT_FEATURE_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"model artifact directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--positive-threshold",
        type=_positive_threshold,
        default=0.5,
        help="mean literature score defining the similar class (default: 0.5)",
    )
    parser.add_argument(
        "--c-grid",
        type=_positive_float,
        nargs="+",
        default=list(DEFAULT_C_GRID),
        help="candidate inverse regularization strengths",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="logistic-regression random seed (default: 42)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    split_dir = args.split_dir.expanduser().resolve()
    feature_dir = args.feature_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    try:
        manifest_path = split_dir / "split_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Split manifest not found: {manifest_path}. Run splits.py first."
            )
        with manifest_path.open("r", encoding="utf-8") as file:
            split_manifest = json.load(file)
        if split_manifest.get("integrity", {}).get("pair_overlap_count") != 0:
            raise ValueError("Split manifest reports ORPHA-pair leakage.")

        targets = {
            split_name: load_pair_targets(
                split_dir / f"{split_name}.csv",
                split_name,
            )
            for split_name in ("train", "validation", "test")
        }
        split_pair_ids = {
            split_name: {target["pair_id"] for target in split_targets}
            for split_name, split_targets in targets.items()
        }
        if (
            split_pair_ids["train"] & split_pair_ids["validation"]
            or split_pair_ids["train"] & split_pair_ids["test"]
            or split_pair_ids["validation"] & split_pair_ids["test"]
        ):
            raise ValueError("An ORPHA pair occurs in more than one split.")

        evaluator = DiseaseDistanceEvaluator(feature_dir=feature_dir)
        datasets = {
            split_name: build_dataset(
                split_targets,
                evaluator,
                positive_threshold=args.positive_threshold,
            )
            for split_name, split_targets in targets.items()
        }
        train_features, train_labels, train_metadata = datasets["train"]
        validation_features, validation_labels, validation_metadata = datasets[
            "validation"
        ]
        test_features, test_labels, test_metadata = datasets["test"]
        _validate_binary_training_labels(train_labels)

        selected_c, validation_model, tuning_trials = select_regularization(
            train_features,
            train_labels,
            validation_features,
            validation_labels,
            c_grid=sorted(set(args.c_grid)),
            random_state=args.random_state,
        )
        validation_evaluation = evaluate_classifier(
            validation_model,
            validation_features,
            validation_labels,
            probability_threshold=0.5,
        )
        validation_metrics = validation_evaluation.metrics
        validation_probabilities = (
            validation_evaluation.positive_probabilities
        )
        if validation_probabilities is None:
            raise RuntimeError(
                "Logistic regression did not return class probabilities."
            )

        final_features = np.vstack([train_features, validation_features])
        final_labels = np.concatenate([train_labels, validation_labels])
        final_model = make_pipeline(selected_c, args.random_state)
        final_model.fit(final_features, final_labels)
        test_evaluation = evaluate_classifier(
            final_model,
            test_features,
            test_labels,
            probability_threshold=0.5,
        )
        test_metrics = test_evaluation.metrics
        test_probabilities = test_evaluation.positive_probabilities
        if test_probabilities is None:
            raise RuntimeError(
                "Logistic regression did not return class probabilities."
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        artifact = {
            "model": final_model,
            "feature_names": list(FEATURE_NAMES),
            "feature_families": list(FEATURE_FAMILIES),
            "positive_threshold": args.positive_threshold,
            "probability_threshold": 0.5,
            "selected_c": selected_c,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "training_pairs": int(len(final_labels)),
            "split_manifest_sha256": _sha256(manifest_path),
        }
        _dump_model_atomic(output_dir / MODEL_FILENAME, artifact)

        predictions = [
            *_prediction_rows(
                "validation",
                validation_metadata,
                validation_probabilities,
                validation_evaluation.predictions,
            ),
            *_prediction_rows(
                "test",
                test_metadata,
                test_probabilities,
                test_evaluation.predictions,
            ),
        ]
        prediction_columns = [
            "split",
            "pair_id",
            "orpha_id_a",
            "orpha_id_b",
            "disease_a",
            "disease_b",
            "literature_similarity",
            "literature_similarity_min",
            "literature_similarity_max",
            "literature_observations",
            "unique_papers",
            "source_rows",
            "relationships",
            "true_label",
            "predicted_label",
            "probability_similar",
            "heuristic_similarity",
        ]
        _write_csv_atomic(
            output_dir / PREDICTIONS_FILENAME,
            prediction_columns,
            predictions,
        )
        coefficient_rows = _coefficient_rows(final_model)
        _write_csv_atomic(
            output_dir / COEFFICIENTS_FILENAME,
            [
                "feature",
                "standardized_coefficient",
                "raw_unit_coefficient",
                "odds_ratio_per_raw_unit",
            ],
            coefficient_rows,
        )

        metrics = {
            "schema_version": "1.0.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": {
                "type": "standardized_l2_logistic_regression",
                "selected_c": selected_c,
                "positive_threshold": args.positive_threshold,
                "probability_threshold": 0.5,
                "class_weight": "balanced",
                "feature_names": list(FEATURE_NAMES),
            },
            "data": {
                "split_directory": str(split_dir),
                "feature_directory": str(feature_dir),
                "split_manifest_sha256": _sha256(manifest_path),
                "pair_counts": {
                    split_name: len(split_targets)
                    for split_name, split_targets in targets.items()
                },
            },
            "validation": validation_metrics,
            "test": test_metrics,
            "regularization_trials": tuning_trials,
            "notes": [
                "Paper scores are averaged per unordered ORPHA pair.",
                "Validation selected regularization; test data was used once.",
                "This classifier is experimental and is not a clinical tool.",
            ],
        }
        _write_json_atomic(output_dir / METRICS_FILENAME, metrics)
    except (
        FileNotFoundError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
        OSError,
        json.JSONDecodeError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"Trained linear classifier in {output_dir}")
    print(
        f"  pairs: {len(train_metadata):,} train, "
        f"{len(validation_metadata):,} validation, {len(test_metadata):,} test"
    )
    print(f"  selected C: {selected_c:g}")
    print(
        "  validation: balanced accuracy "
        f"{validation_metrics['balanced_accuracy']:.3f}, "
        f"ROC AUC {validation_metrics['roc_auc']:.3f}"
    )
    print(
        "  test: balanced accuracy "
        f"{test_metrics['balanced_accuracy']:.3f}, "
        f"ROC AUC {test_metrics['roc_auc']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
