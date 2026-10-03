"""Train linear regression to predict literature similarity scores.

The data preparation and pair features intentionally match
``linear_classifier.py``.  Paper scores are averaged per unordered ORPHA pair,
but remain continuous targets in [0, 1].  Ridge regularization is selected by
validation RMSE before the model is refit on train+validation and evaluated on
the untouched test split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import DEFAULT_FEATURE_DIR, DiseaseDistanceEvaluator  # noqa: E402
from models.evaluation import (  # noqa: E402
    evaluate_regressor,
    regression_metrics,
)
from models.linear_classifier import (  # noqa: E402
    FEATURE_FAMILIES,
    FEATURE_NAMES,
    extract_pair_features,
    load_pair_targets,
)


DEFAULT_SPLIT_DIR = PROJECT_ROOT / ".data" / "splits"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".data" / "models" / "linear_regression"
DEFAULT_ALPHA_GRID = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
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


def build_dataset(
    targets: Sequence[Mapping[str, Any]],
    evaluator: DiseaseDistanceEvaluator,
) -> tuple[np.ndarray, np.ndarray, list[Dict[str, Any]]]:
    """Join continuous pair targets to the generated pairwise features."""

    feature_rows: list[np.ndarray] = []
    scores: list[float] = []
    metadata: list[Dict[str, Any]] = []

    for target in targets:
        disease_a_id, disease_b_id = _parse_pair_id(str(target["pair_id"]))
        features, result = extract_pair_features(
            evaluator,
            disease_a_id,
            disease_b_id,
        )
        score = float(target["literature_similarity"])
        feature_rows.append(features)
        scores.append(score)
        metadata.append(
            {
                **target,
                "orpha_id_a": disease_a_id,
                "orpha_id_b": disease_b_id,
                "disease_a": result["disease_a"]["name"],
                "disease_b": result["disease_b"]["name"],
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
        np.asarray(scores, dtype=np.float64),
        metadata,
    )


def make_pipeline(alpha: float) -> Pipeline:
    """Create a standardized ridge-regression pipeline."""

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=alpha)),
        ]
    )


def select_regularization(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    *,
    alpha_grid: Sequence[float],
) -> tuple[float, Pipeline, list[Dict[str, Any]]]:
    """Select alpha by validation RMSE, then MAE."""

    trials: list[Dict[str, Any]] = []
    best_key: Optional[tuple[float, float, float]] = None
    best_alpha = 0.0
    best_model: Optional[Pipeline] = None

    for alpha in alpha_grid:
        model = make_pipeline(alpha)
        model.fit(train_features, train_targets)
        evaluation = evaluate_regressor(
            model,
            validation_features,
            validation_targets,
        )
        metrics = evaluation.metrics
        clipped_metrics = regression_metrics(
            validation_targets,
            np.clip(evaluation.predictions, 0.0, 1.0),
        )
        trials.append(
            {
                "alpha": alpha,
                **metrics,
                "clipped_to_unit_interval": clipped_metrics,
            }
        )
        # If errors tie, prefer stronger regularization.
        selection_key = (
            float(metrics["root_mean_squared_error"]),
            float(metrics["mean_absolute_error"]),
            -alpha,
        )
        if best_key is None or selection_key < best_key:
            best_key = selection_key
            best_alpha = alpha
            best_model = model

    if best_model is None:
        raise ValueError("The alpha grid is empty.")
    return best_alpha, best_model, trials


def _prediction_rows(
    split_name: str,
    metadata: Sequence[Mapping[str, Any]],
    predictions: np.ndarray,
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for item, prediction in zip(metadata, predictions):
        target = float(item["literature_similarity"])
        raw_prediction = float(prediction)
        clipped_prediction = min(1.0, max(0.0, raw_prediction))
        rows.append(
            {
                "split": split_name,
                "pair_id": item["pair_id"],
                "orpha_id_a": item["orpha_id_a"],
                "orpha_id_b": item["orpha_id_b"],
                "disease_a": item["disease_a"],
                "disease_b": item["disease_b"],
                "literature_similarity": target,
                "literature_similarity_min": item["literature_similarity_min"],
                "literature_similarity_max": item["literature_similarity_max"],
                "literature_observations": item["literature_observations"],
                "unique_papers": item["unique_papers"],
                "source_rows": item["source_rows"],
                "relationships": item["relationships"],
                "predicted_similarity": raw_prediction,
                "predicted_similarity_clipped": clipped_prediction,
                "absolute_error": abs(raw_prediction - target),
                "clipped_absolute_error": abs(clipped_prediction - target),
                "heuristic_similarity": item["heuristic_similarity"],
            }
        )
    return rows


def _coefficient_rows(model: Pipeline) -> tuple[list[Dict[str, Any]], float]:
    scaler: StandardScaler = model.named_steps["scaler"]
    regressor: Ridge = model.named_steps["regressor"]
    standardized = np.asarray(regressor.coef_, dtype=np.float64)
    raw_unit = standardized / scaler.scale_
    raw_intercept = float(
        regressor.intercept_ - np.sum(standardized * scaler.mean_ / scaler.scale_)
    )
    rows = [
        {
            "feature": feature_name,
            "standardized_coefficient": float(standardized_coefficient),
            "raw_unit_coefficient": float(raw_coefficient),
        }
        for feature_name, standardized_coefficient, raw_coefficient in zip(
            FEATURE_NAMES,
            standardized,
            raw_unit,
        )
    ]
    return (
        sorted(
            rows,
            key=lambda row: abs(row["standardized_coefficient"]),
            reverse=True,
        ),
        raw_intercept,
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


def _nonnegative_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be a number") from error
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
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
        "--alpha-grid",
        type=_nonnegative_float,
        nargs="+",
        default=list(DEFAULT_ALPHA_GRID),
        help="candidate ridge regularization strengths, including 0 for OLS",
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
        pair_ids = {
            split_name: {target["pair_id"] for target in split_targets}
            for split_name, split_targets in targets.items()
        }
        if (
            pair_ids["train"] & pair_ids["validation"]
            or pair_ids["train"] & pair_ids["test"]
            or pair_ids["validation"] & pair_ids["test"]
        ):
            raise ValueError("An ORPHA pair occurs in more than one split.")

        evaluator = DiseaseDistanceEvaluator(feature_dir=feature_dir)
        datasets = {
            split_name: build_dataset(split_targets, evaluator)
            for split_name, split_targets in targets.items()
        }
        train_features, train_scores, train_metadata = datasets["train"]
        validation_features, validation_scores, validation_metadata = datasets[
            "validation"
        ]
        test_features, test_scores, test_metadata = datasets["test"]

        selected_alpha, validation_model, tuning_trials = select_regularization(
            train_features,
            train_scores,
            validation_features,
            validation_scores,
            alpha_grid=sorted(set(args.alpha_grid)),
        )
        validation_evaluation = evaluate_regressor(
            validation_model,
            validation_features,
            validation_scores,
        )

        final_features = np.vstack([train_features, validation_features])
        final_scores = np.concatenate([train_scores, validation_scores])
        final_model = make_pipeline(selected_alpha)
        final_model.fit(final_features, final_scores)
        test_evaluation = evaluate_regressor(
            final_model,
            test_features,
            test_scores,
        )

        validation_clipped_metrics = regression_metrics(
            validation_scores,
            np.clip(validation_evaluation.predictions, 0.0, 1.0),
        )
        test_clipped_metrics = regression_metrics(
            test_scores,
            np.clip(test_evaluation.predictions, 0.0, 1.0),
        )
        validation_baseline = regression_metrics(
            validation_scores,
            np.full_like(validation_scores, np.mean(train_scores)),
        )
        test_baseline = regression_metrics(
            test_scores,
            np.full_like(test_scores, np.mean(final_scores)),
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        coefficient_rows, raw_intercept = _coefficient_rows(final_model)
        artifact = {
            "model": final_model,
            "feature_names": list(FEATURE_NAMES),
            "feature_families": list(FEATURE_FAMILIES),
            "target": "paper_averaged_similarity_score",
            "selected_alpha": selected_alpha,
            "raw_unit_intercept": raw_intercept,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "training_pairs": int(len(final_scores)),
            "split_manifest_sha256": _sha256(manifest_path),
            "prediction_note": (
                "Ridge predictions are unbounded; clip to [0, 1] when a "
                "bounded display score is required."
            ),
        }
        _dump_model_atomic(output_dir / MODEL_FILENAME, artifact)

        predictions = [
            *_prediction_rows(
                "validation",
                validation_metadata,
                validation_evaluation.predictions,
            ),
            *_prediction_rows(
                "test",
                test_metadata,
                test_evaluation.predictions,
            ),
        ]
        _write_csv_atomic(
            output_dir / PREDICTIONS_FILENAME,
            [
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
                "predicted_similarity",
                "predicted_similarity_clipped",
                "absolute_error",
                "clipped_absolute_error",
                "heuristic_similarity",
            ],
            predictions,
        )
        _write_csv_atomic(
            output_dir / COEFFICIENTS_FILENAME,
            [
                "feature",
                "standardized_coefficient",
                "raw_unit_coefficient",
            ],
            coefficient_rows,
        )

        metrics = {
            "schema_version": "1.0.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": {
                "type": "standardized_ridge_regression",
                "selected_alpha": selected_alpha,
                "target": "paper_averaged_similarity_score",
                "feature_names": list(FEATURE_NAMES),
                "raw_unit_intercept": raw_intercept,
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
            "validation": {
                "raw_predictions": validation_evaluation.metrics,
                "clipped_to_unit_interval": validation_clipped_metrics,
                "mean_target_baseline": validation_baseline,
            },
            "test": {
                "raw_predictions": test_evaluation.metrics,
                "clipped_to_unit_interval": test_clipped_metrics,
                "mean_target_baseline": test_baseline,
            },
            "regularization_trials": tuning_trials,
            "notes": [
                "Each paper gets one vote before scores are averaged by pair.",
                "Validation RMSE selected alpha; test data was used once.",
                "Raw linear predictions may fall outside [0, 1].",
                "This model is experimental and is not a clinical tool.",
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

    validation_metrics = validation_evaluation.metrics
    test_metrics = test_evaluation.metrics
    print(f"Trained linear regression in {output_dir}")
    print(
        f"  pairs: {len(train_metadata):,} train, "
        f"{len(validation_metadata):,} validation, {len(test_metadata):,} test"
    )
    print(f"  selected alpha: {selected_alpha:g}")
    print(
        "  validation: RMSE "
        f"{validation_metrics['root_mean_squared_error']:.3f}, "
        f"MAE {validation_metrics['mean_absolute_error']:.3f}"
    )
    print(
        "  test: RMSE "
        f"{test_metrics['root_mean_squared_error']:.3f}, "
        f"MAE {test_metrics['mean_absolute_error']:.3f}, "
        f"R2 {test_metrics['r2']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
