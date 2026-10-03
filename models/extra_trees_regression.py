"""Train Extra Trees regression for literature disease-similarity scores.

This nonlinear baseline uses the same paper-averaged targets and symmetric
pair features as the linear models.  Tree depth, feature subsampling, and leaf
size are selected on validation RMSE.  The selected architecture is then
refit on train+validation and evaluated once on the test split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import DEFAULT_FEATURE_DIR, DiseaseDistanceEvaluator  # noqa: E402
from models.evaluation import evaluate_regressor, regression_metrics  # noqa: E402
from models.linear_classifier import (  # noqa: E402
    FEATURE_NAMES,
    load_pair_targets,
)
from models.linear_regression import build_dataset  # noqa: E402


DEFAULT_SPLIT_DIR = PROJECT_ROOT / ".data" / "splits"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".data" / "models" / "extra_trees_regression"
DEFAULT_MAX_FEATURES = ("sqrt", 0.25, 0.5, 1.0)
DEFAULT_MIN_SAMPLES_LEAF = (1, 2, 4)
DEFAULT_MAX_DEPTH = (None, 12)
MODEL_FILENAME = "model.joblib"
METRICS_FILENAME = "metrics.json"
IMPORTANCES_FILENAME = "feature_importances.csv"
PREDICTIONS_FILENAME = "predictions.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_model(
    *,
    n_estimators: int,
    max_features: str | float,
    min_samples_leaf: int,
    max_depth: Optional[int],
    random_state: int,
) -> ExtraTreesRegressor:
    """Create one deterministic, parallel Extra Trees candidate."""

    return ExtraTreesRegressor(
        n_estimators=n_estimators,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )


def select_hyperparameters(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    *,
    n_estimators: int,
    random_state: int,
) -> tuple[Dict[str, Any], ExtraTreesRegressor, list[Dict[str, Any]]]:
    """Select tree complexity by validation RMSE, then MAE."""

    trials: list[Dict[str, Any]] = []
    best_key: Optional[tuple[float, float, int, int]] = None
    best_parameters: Optional[Dict[str, Any]] = None
    best_model: Optional[ExtraTreesRegressor] = None

    candidates = itertools.product(
        DEFAULT_MAX_FEATURES,
        DEFAULT_MIN_SAMPLES_LEAF,
        DEFAULT_MAX_DEPTH,
    )
    for max_features, min_samples_leaf, max_depth in candidates:
        model = make_model(
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            random_state=random_state,
        )
        model.fit(train_features, train_targets)
        evaluation = evaluate_regressor(
            model,
            validation_features,
            validation_targets,
        )
        metrics = evaluation.metrics
        parameters = {
            "max_features": max_features,
            "min_samples_leaf": min_samples_leaf,
            "max_depth": max_depth,
        }
        trials.append({**parameters, **metrics})

        # Prefer simpler trees when predictive errors are tied.
        selection_key = (
            float(metrics["root_mean_squared_error"]),
            float(metrics["mean_absolute_error"]),
            -min_samples_leaf,
            max_depth if max_depth is not None else sys.maxsize,
        )
        if best_key is None or selection_key < best_key:
            best_key = selection_key
            best_parameters = parameters
            best_model = model

    if best_model is None or best_parameters is None:
        raise RuntimeError("No Extra Trees hyperparameter candidates were evaluated.")
    return best_parameters, best_model, trials


def _prediction_rows(
    split_name: str,
    metadata: Sequence[Mapping[str, Any]],
    predictions: np.ndarray,
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for item, prediction in zip(metadata, predictions):
        target = float(item["literature_similarity"])
        predicted = float(prediction)
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
                "relationships": item["relationships"],
                "predicted_similarity": predicted,
                "absolute_error": abs(predicted - target),
                "heuristic_similarity": item["heuristic_similarity"],
            }
        )
    return rows


def _importance_rows(model: ExtraTreesRegressor) -> list[Dict[str, Any]]:
    rows = [
        {
            "feature": feature,
            "importance": float(importance),
        }
        for feature, importance in zip(FEATURE_NAMES, model.feature_importances_)
    ]
    return sorted(rows, key=lambda row: row["importance"], reverse=True)


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


def _positive_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
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
        "--n-estimators",
        type=_positive_integer,
        default=300,
        help="trees per validation candidate and final model (default: 300)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="random seed (default: 42)",
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

        selected_parameters, validation_model, trials = select_hyperparameters(
            train_features,
            train_scores,
            validation_features,
            validation_scores,
            n_estimators=args.n_estimators,
            random_state=args.random_state,
        )
        validation_evaluation = evaluate_regressor(
            validation_model,
            validation_features,
            validation_scores,
        )

        final_features = np.vstack([train_features, validation_features])
        final_scores = np.concatenate([train_scores, validation_scores])
        final_model = make_model(
            n_estimators=args.n_estimators,
            random_state=args.random_state,
            **selected_parameters,
        )
        final_model.fit(final_features, final_scores)
        test_evaluation = evaluate_regressor(
            final_model,
            test_features,
            test_scores,
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
        artifact = {
            "model": final_model,
            "feature_names": list(FEATURE_NAMES),
            "target": "paper_averaged_similarity_score",
            "selected_parameters": selected_parameters,
            "n_estimators": args.n_estimators,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "training_pairs": int(len(final_scores)),
            "split_manifest_sha256": _sha256(manifest_path),
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
                "relationships",
                "predicted_similarity",
                "absolute_error",
                "heuristic_similarity",
            ],
            predictions,
        )
        _write_csv_atomic(
            output_dir / IMPORTANCES_FILENAME,
            ["feature", "importance"],
            _importance_rows(final_model),
        )

        metrics = {
            "schema_version": "1.0.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": {
                "type": "extra_trees_regression",
                "target": "paper_averaged_similarity_score",
                "n_estimators": args.n_estimators,
                "selected_parameters": selected_parameters,
                "feature_count": len(FEATURE_NAMES),
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
                "model": validation_evaluation.metrics,
                "mean_target_baseline": validation_baseline,
            },
            "test": {
                "model": test_evaluation.metrics,
                "mean_target_baseline": test_baseline,
            },
            "hyperparameter_trials": trials,
            "notes": [
                "Each paper gets one vote before scores are averaged by pair.",
                "Validation RMSE selected tree complexity; test data was used once.",
                "Impurity importance can favor continuous or high-variance features.",
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
    print(f"Trained Extra Trees regression in {output_dir}")
    print(
        f"  pairs: {len(train_metadata):,} train, "
        f"{len(validation_metadata):,} validation, {len(test_metadata):,} test"
    )
    print(
        "  selected: "
        f"max_features={selected_parameters['max_features']}, "
        f"min_samples_leaf={selected_parameters['min_samples_leaf']}, "
        f"max_depth={selected_parameters['max_depth']}"
    )
    print(
        "  validation: RMSE "
        f"{validation_metrics['root_mean_squared_error']:.3f}, "
        f"R2 {validation_metrics['r2']:.3f}"
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
