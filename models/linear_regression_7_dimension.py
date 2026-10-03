"""Train confidence-weighted linear models for seven similarity dimensions.

For each dimension, all paper scores for the same disease pair are collapsed
into one confidence-weighted consensus target.  The sum of the contributing
paper weights is passed to ridge regression as the pair's sample weight.  Every
dimension receives a biologically relevant subset of the shared symmetric
pair-feature representation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
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
    evaluate_dimension_regressors,
    weighted_regression_metrics,
)
from models.linear_classifier import (  # noqa: E402
    FEATURE_NAMES,
    extract_pair_features,
)


DEFAULT_SPLIT_DIR = PROJECT_ROOT / ".data" / "splits" / "dimensions"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / ".data" / "models" / "linear_regression_7_dimension"
)
DEFAULT_ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
DIMENSION_FEATURE_FAMILIES: Dict[str, tuple[str, ...]] = {
    "phenotype": ("phenotypes", "body_systems", "onset"),
    "genetic": ("genes", "inheritance"),
    "mechanism": ("genes", "classifications", "body_systems"),
    "therapeutic": ("approved_drugs", "genes"),
    "natural_history": ("phenotypes", "onset", "prevalence"),
    "diagnostic_confusability": (
        "phenotypes",
        "body_systems",
        "onset",
        "classifications",
    ),
    "comorbidity": ("body_systems", "classifications", "phenotypes"),
}
DIMENSIONS = tuple(DIMENSION_FEATURE_FAMILIES)
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


def dimension_feature_indices() -> Dict[str, np.ndarray]:
    """Return stable column selections for every biological dimension."""

    selections: Dict[str, np.ndarray] = {}
    for dimension, families in DIMENSION_FEATURE_FAMILIES.items():
        indices = [
            index
            for index, feature_name in enumerate(FEATURE_NAMES)
            if any(
                feature_name.startswith(f"{family}_")
                for family in families
            )
        ]
        if not indices:
            raise RuntimeError(f"No model features selected for {dimension}.")
        selections[dimension] = np.asarray(indices, dtype=np.int64)
    return selections


def load_dimension_datasets(
    split_dir: Path,
    evaluator: DiseaseDistanceEvaluator,
) -> tuple[
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, Dict[str, list[Dict[str, Any]]]],
]:
    """Load splits and aggregate paper scores into pair-level targets."""

    selections = dimension_feature_indices()
    feature_cache: Dict[str, np.ndarray] = {}
    aggregates: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {
        split_name: {
            dimension: {}
            for dimension in DIMENSIONS
        }
        for split_name in ("train", "validation", "test")
    }

    for split_name in ("train", "validation", "test"):
        path = split_dir / f"{split_name}.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"Dimension split not found: {path}. Run splits.py first."
            )
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            required = {
                "_pair_id",
                "_paper_id",
                "_split",
                "disease_a",
                "disease_b",
                *(f"w_{dimension}" for dimension in DIMENSIONS),
                *(f"s_{dimension}" for dimension in DIMENSIONS),
            }
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"{path} is missing required column(s): "
                    + ", ".join(sorted(missing))
                )

            for row_number, row in enumerate(reader, start=2):
                if row["_split"] != split_name:
                    raise ValueError(
                        f"{path}:{row_number} declares split "
                        f"{row['_split']!r}; expected {split_name!r}."
                    )
                pair_id = row["_pair_id"].strip()
                disease_a_id, disease_b_id = _parse_pair_id(pair_id)
                if pair_id not in feature_cache:
                    full_features, _ = extract_pair_features(
                        evaluator,
                        disease_a_id,
                        disease_b_id,
                    )
                    feature_cache[pair_id] = full_features

                for dimension in DIMENSIONS:
                    raw_score = row[f"s_{dimension}"].strip()
                    weight = float(row[f"w_{dimension}"])
                    if not raw_score or weight <= 0:
                        continue
                    score = float(raw_score)
                    pair_aggregates = aggregates[split_name][dimension]
                    aggregate = pair_aggregates.get(pair_id)
                    if aggregate is None:
                        aggregate = {
                            "split": split_name,
                            "dimension": dimension,
                            "pair_id": pair_id,
                            "disease_a": row["disease_a"],
                            "disease_b": row["disease_b"],
                            "orpha_id_a": disease_a_id,
                            "orpha_id_b": disease_b_id,
                            "features": feature_cache[pair_id][
                                selections[dimension]
                            ],
                            "weighted_score_sum": 0.0,
                            "weighted_squared_score_sum": 0.0,
                            "confidence_weight": 0.0,
                            "paper_scores": {},
                            "pmids": set(),
                        }
                        pair_aggregates[pair_id] = aggregate

                    paper_id = row["_paper_id"].strip()
                    prior = aggregate["paper_scores"].get(paper_id)
                    paper_value = (score, weight)
                    if prior is not None:
                        if prior != paper_value:
                            raise ValueError(
                                f"{path}:{row_number} has conflicting duplicate "
                                f"score for {paper_id}, {pair_id}, {dimension}."
                            )
                        continue

                    aggregate["paper_scores"][paper_id] = paper_value
                    pmid = row.get("pmid", "").strip()
                    if pmid:
                        aggregate["pmids"].add(pmid)
                    aggregate["weighted_score_sum"] += weight * score
                    aggregate["weighted_squared_score_sum"] += (
                        weight * score * score
                    )
                    aggregate["confidence_weight"] += weight

    arrays: Dict[str, Dict[str, np.ndarray]] = {}
    metadata: Dict[str, Dict[str, list[Dict[str, Any]]]] = {
        split_name: {dimension: [] for dimension in DIMENSIONS}
        for split_name in aggregates
    }
    for split_name in aggregates:
        arrays[split_name] = {}
        for dimension in DIMENSIONS:
            dimension_aggregates = aggregates[split_name][dimension]
            if not dimension_aggregates:
                raise ValueError(
                    f"No positive-weight {dimension} scores in {split_name} split."
                )
            feature_rows: list[np.ndarray] = []
            target_rows: list[float] = []
            weight_rows: list[float] = []
            for pair_id in sorted(dimension_aggregates):
                aggregate = dimension_aggregates[pair_id]
                confidence_weight = float(aggregate["confidence_weight"])
                consensus_score = (
                    float(aggregate["weighted_score_sum"])
                    / confidence_weight
                )
                weighted_variance = max(
                    0.0,
                    (
                        float(aggregate["weighted_squared_score_sum"])
                        / confidence_weight
                    )
                    - consensus_score * consensus_score,
                )
                if weighted_variance < 1e-15:
                    weighted_variance = 0.0
                paper_scores = aggregate["paper_scores"]
                raw_scores = [value[0] for value in paper_scores.values()]

                feature_rows.append(aggregate["features"])
                target_rows.append(consensus_score)
                weight_rows.append(confidence_weight)
                metadata[split_name][dimension].append(
                    {
                        "split": aggregate["split"],
                        "dimension": aggregate["dimension"],
                        "pair_id": pair_id,
                        "paper_ids": ";".join(sorted(paper_scores)),
                        "pmids": ";".join(sorted(aggregate["pmids"])),
                        "paper_count": len(paper_scores),
                        "disease_a": aggregate["disease_a"],
                        "disease_b": aggregate["disease_b"],
                        "orpha_id_a": aggregate["orpha_id_a"],
                        "orpha_id_b": aggregate["orpha_id_b"],
                        "similarity_score": consensus_score,
                        "confidence_weight": confidence_weight,
                        "source_score_min": min(raw_scores),
                        "source_score_max": max(raw_scores),
                        "source_score_weighted_std": math.sqrt(
                            weighted_variance
                        ),
                    }
                )

            arrays[split_name][f"{dimension}_features"] = np.vstack(
                feature_rows
            )
            arrays[split_name][f"{dimension}_targets"] = np.asarray(
                target_rows,
                dtype=np.float64,
            )
            arrays[split_name][f"{dimension}_weights"] = np.asarray(
                weight_rows,
                dtype=np.float64,
            )
    return arrays, metadata


def make_pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=alpha)),
        ]
    )


def fit_weighted(
    model: Pipeline,
    features: np.ndarray,
    targets: np.ndarray,
    sample_weights: np.ndarray,
) -> Pipeline:
    """Fit ridge regression using aggregate pair confidence weights."""

    model.fit(
        features,
        targets,
        regressor__sample_weight=sample_weights,
    )
    return model


def select_regularization(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    train_weights: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    validation_weights: np.ndarray,
) -> tuple[float, Pipeline, list[Dict[str, Any]]]:
    """Select ridge alpha by confidence-weighted validation RMSE."""

    trials: list[Dict[str, Any]] = []
    best_key: Optional[tuple[float, float, float]] = None
    best_alpha = 0.0
    best_model: Optional[Pipeline] = None
    for alpha in DEFAULT_ALPHA_GRID:
        model = fit_weighted(
            make_pipeline(alpha),
            train_features,
            train_targets,
            train_weights,
        )
        predictions = model.predict(validation_features)
        metrics = weighted_regression_metrics(
            validation_targets,
            predictions,
            validation_weights,
        )
        trials.append({"alpha": alpha, **metrics})
        key = (
            float(metrics["root_mean_squared_error"]),
            float(metrics["mean_absolute_error"]),
            -alpha,
        )
        if best_key is None or key < best_key:
            best_key = key
            best_alpha = alpha
            best_model = model
    if best_model is None:
        raise RuntimeError("No regularization candidates were fitted.")
    return best_alpha, best_model, trials


def _dimension_maps(
    arrays: Mapping[str, np.ndarray],
    suffix: str,
) -> Dict[str, np.ndarray]:
    return {
        dimension: arrays[f"{dimension}_{suffix}"]
        for dimension in DIMENSIONS
    }


def _coefficient_rows(
    models: Mapping[str, Pipeline],
    selections: Mapping[str, np.ndarray],
) -> tuple[list[Dict[str, Any]], Dict[str, float]]:
    rows: list[Dict[str, Any]] = []
    intercepts: Dict[str, float] = {}
    for dimension, model in models.items():
        scaler: StandardScaler = model.named_steps["scaler"]
        regressor: Ridge = model.named_steps["regressor"]
        standardized = np.asarray(regressor.coef_, dtype=np.float64)
        raw = standardized / scaler.scale_
        intercepts[dimension] = float(
            regressor.intercept_
            - np.sum(standardized * scaler.mean_ / scaler.scale_)
        )
        names = [FEATURE_NAMES[index] for index in selections[dimension]]
        rows.extend(
            {
                "dimension": dimension,
                "feature": name,
                "standardized_coefficient": float(standardized_value),
                "raw_unit_coefficient": float(raw_value),
            }
            for name, standardized_value, raw_value in zip(
                names,
                standardized,
                raw,
            )
        )
    rows.sort(
        key=lambda row: (
            row["dimension"],
            -abs(row["standardized_coefficient"]),
        )
    )
    return rows, intercepts


def _prediction_rows(
    metadata: Mapping[str, Sequence[Mapping[str, Any]]],
    evaluations: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for dimension in DIMENSIONS:
        for item, prediction in zip(
            metadata[dimension],
            evaluations[dimension].predictions,
        ):
            predicted = float(prediction)
            target = float(item["similarity_score"])
            rows.append(
                {
                    **item,
                    "predicted_similarity": predicted,
                    "predicted_similarity_clipped": min(
                        1.0,
                        max(0.0, predicted),
                    ),
                    "absolute_error": abs(predicted - target),
                    "weighted_absolute_error": (
                        abs(predicted - target)
                        * float(item["confidence_weight"])
                    ),
                }
            )
    return rows


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


def _dump_model_atomic(path: Path, artifact: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        joblib.dump(artifact, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=DEFAULT_SPLIT_DIR,
        help=f"seven-dimension split directory (default: {DEFAULT_SPLIT_DIR})",
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
                f"Dimension split manifest not found: {manifest_path}. "
                "Run splits.py first."
            )
        with manifest_path.open("r", encoding="utf-8") as file:
            split_manifest = json.load(file)
        if tuple(split_manifest.get("dimensions", ())) != DIMENSIONS:
            raise ValueError("Dimension split manifest has unexpected dimensions.")
        if split_manifest.get("integrity", {}).get("pair_overlap_count") != 0:
            raise ValueError("Dimension split manifest reports ORPHA-pair leakage.")

        evaluator = DiseaseDistanceEvaluator(feature_dir=feature_dir)
        arrays, metadata = load_dimension_datasets(split_dir, evaluator)
        selections = dimension_feature_indices()
        selected_alphas: Dict[str, float] = {}
        validation_models: Dict[str, Pipeline] = {}
        trials: Dict[str, list[Dict[str, Any]]] = {}

        for dimension in DIMENSIONS:
            (
                selected_alphas[dimension],
                validation_models[dimension],
                trials[dimension],
            ) = select_regularization(
                arrays["train"][f"{dimension}_features"],
                arrays["train"][f"{dimension}_targets"],
                arrays["train"][f"{dimension}_weights"],
                arrays["validation"][f"{dimension}_features"],
                arrays["validation"][f"{dimension}_targets"],
                arrays["validation"][f"{dimension}_weights"],
            )

        validation_evaluations = evaluate_dimension_regressors(
            validation_models,
            _dimension_maps(arrays["validation"], "features"),
            _dimension_maps(arrays["validation"], "targets"),
            _dimension_maps(arrays["validation"], "weights"),
        )

        final_models: Dict[str, Pipeline] = {}
        for dimension in DIMENSIONS:
            final_models[dimension] = fit_weighted(
                make_pipeline(selected_alphas[dimension]),
                np.vstack(
                    [
                        arrays["train"][f"{dimension}_features"],
                        arrays["validation"][f"{dimension}_features"],
                    ]
                ),
                np.concatenate(
                    [
                        arrays["train"][f"{dimension}_targets"],
                        arrays["validation"][f"{dimension}_targets"],
                    ]
                ),
                np.concatenate(
                    [
                        arrays["train"][f"{dimension}_weights"],
                        arrays["validation"][f"{dimension}_weights"],
                    ]
                ),
            )

        test_evaluations = evaluate_dimension_regressors(
            final_models,
            _dimension_maps(arrays["test"], "features"),
            _dimension_maps(arrays["test"], "targets"),
            _dimension_maps(arrays["test"], "weights"),
        )

        validation_baselines: Dict[str, Dict[str, Any]] = {}
        test_baselines: Dict[str, Dict[str, Any]] = {}
        for dimension in DIMENSIONS:
            train_targets = arrays["train"][f"{dimension}_targets"]
            train_weights = arrays["train"][f"{dimension}_weights"]
            train_mean = float(np.average(train_targets, weights=train_weights))
            final_targets = np.concatenate(
                [
                    train_targets,
                    arrays["validation"][f"{dimension}_targets"],
                ]
            )
            final_weights = np.concatenate(
                [
                    train_weights,
                    arrays["validation"][f"{dimension}_weights"],
                ]
            )
            final_mean = float(np.average(final_targets, weights=final_weights))
            validation_baselines[dimension] = weighted_regression_metrics(
                arrays["validation"][f"{dimension}_targets"],
                np.full_like(
                    arrays["validation"][f"{dimension}_targets"],
                    train_mean,
                ),
                arrays["validation"][f"{dimension}_weights"],
            )
            test_baselines[dimension] = weighted_regression_metrics(
                arrays["test"][f"{dimension}_targets"],
                np.full_like(
                    arrays["test"][f"{dimension}_targets"],
                    final_mean,
                ),
                arrays["test"][f"{dimension}_weights"],
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        coefficient_rows, raw_intercepts = _coefficient_rows(
            final_models,
            selections,
        )
        _write_csv_atomic(
            output_dir / COEFFICIENTS_FILENAME,
            [
                "dimension",
                "feature",
                "standardized_coefficient",
                "raw_unit_coefficient",
            ],
            coefficient_rows,
        )
        prediction_rows = [
            *_prediction_rows(
                metadata["validation"],
                validation_evaluations,
            ),
            *_prediction_rows(metadata["test"], test_evaluations),
        ]
        _write_csv_atomic(
            output_dir / PREDICTIONS_FILENAME,
            [
                "split",
                "dimension",
                "pair_id",
                "paper_ids",
                "pmids",
                "paper_count",
                "disease_a",
                "disease_b",
                "orpha_id_a",
                "orpha_id_b",
                "similarity_score",
                "confidence_weight",
                "source_score_min",
                "source_score_max",
                "source_score_weighted_std",
                "predicted_similarity",
                "predicted_similarity_clipped",
                "absolute_error",
                "weighted_absolute_error",
            ],
            prediction_rows,
        )

        artifact = {
            "models": final_models,
            "feature_names": {
                dimension: [
                    FEATURE_NAMES[index] for index in selections[dimension]
                ]
                for dimension in DIMENSIONS
            },
            "selected_alphas": selected_alphas,
            "raw_unit_intercepts": raw_intercepts,
            "dimensions": DIMENSIONS,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "split_manifest_sha256": _sha256(manifest_path),
        }
        _dump_model_atomic(output_dir / MODEL_FILENAME, artifact)

        dimension_metrics: Dict[str, Any] = {}
        for dimension in DIMENSIONS:
            dimension_metrics[dimension] = {
                "selected_alpha": selected_alphas[dimension],
                "feature_count": int(len(selections[dimension])),
                "counts": {
                    split_name: int(
                        len(arrays[split_name][f"{dimension}_targets"])
                    )
                    for split_name in ("train", "validation", "test")
                },
                "paper_counts": {
                    split_name: int(
                        sum(
                            row["paper_count"]
                            for row in metadata[split_name][dimension]
                        )
                    )
                    for split_name in ("train", "validation", "test")
                },
                "validation": validation_evaluations[dimension].metrics,
                "validation_mean_baseline": validation_baselines[dimension],
                "test": test_evaluations[dimension].metrics,
                "test_mean_baseline": test_baselines[dimension],
                "regularization_trials": trials[dimension],
            }

        weighted_test_rmses = [
            dimension_metrics[dimension]["test"]["confidence_weighted"][
                "root_mean_squared_error"
            ]
            for dimension in DIMENSIONS
        ]
        metrics = {
            "schema_version": "2.0.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": {
                "type": "seven_confidence_weighted_ridge_regressions",
                "target": (
                    "dimension-specific confidence-weighted mean score per "
                    "disease pair"
                ),
                "dimensions": list(DIMENSIONS),
                "full_pair_feature_count": len(FEATURE_NAMES),
            },
            "data": {
                "split_directory": str(split_dir),
                "feature_directory": str(feature_dir),
                "split_manifest_sha256": _sha256(manifest_path),
            },
            "dimensions": dimension_metrics,
            "summary": {
                "macro_confidence_weighted_test_rmse": float(
                    np.mean(weighted_test_rmses)
                )
            },
            "notes": [
                (
                    "Each target is the confidence-weighted mean of all paper "
                    "scores for that disease pair and dimension."
                ),
                (
                    "The sum of contributing paper confidence weights is used "
                    "as the pair-level sample weight."
                ),
                "Validation selected regularization separately for each dimension.",
                "Test data was used once after train+validation refitting.",
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

    print(f"Trained seven dimension models in {output_dir}")
    for dimension in DIMENSIONS:
        weighted = test_evaluations[dimension].metrics["confidence_weighted"]
        baseline = test_baselines[dimension]
        print(
            f"  {dimension}: n={weighted['examples']}, "
            f"RMSE={weighted['root_mean_squared_error']:.3f} "
            f"(baseline {baseline['root_mean_squared_error']:.3f}), "
            f"R2={weighted['r2']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
