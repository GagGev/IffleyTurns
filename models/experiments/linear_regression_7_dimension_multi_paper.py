"""Train seven-dimension ridge models using only multi-paper disease pairs.

This experiment mirrors ``models/linear_regression_7_dimension.py`` except
that a disease pair may contribute to model fitting only when at least two
distinct papers in its split discuss that pair. Validation and test metrics
are still computed over every scored pair so they remain directly comparable
with the original model.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import DEFAULT_FEATURE_DIR, DiseaseDistanceEvaluator  # noqa: E402
from models import linear_regression_7_dimension as base  # noqa: E402
from models.evaluation import (  # noqa: E402
    evaluate_dimension_regressors,
    weighted_regression_metrics,
)


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / ".data"
    / "models"
    / "experiments"
    / "linear_regression_7_dimension_multi_paper"
)
MIN_PAIR_PAPER_COUNT = 2


def load_pair_paper_counts(
    split_dir: Path,
) -> Dict[str, Dict[str, int]]:
    """Count distinct papers about each pair, independent of dimension."""

    counts: Dict[str, Dict[str, int]] = {}
    for split_name in ("train", "validation", "test"):
        path = split_dir / f"{split_name}.csv"
        papers_by_pair: Dict[str, set[str]] = {}
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            required = {"_pair_id", "_paper_id", "_split"}
            missing = required - set(reader.fieldnames or ())
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
                paper_id = row["_paper_id"].strip()
                if not pair_id or not paper_id:
                    raise ValueError(
                        f"{path}:{row_number} has an empty pair or paper ID."
                    )
                papers_by_pair.setdefault(pair_id, set()).add(paper_id)
        counts[split_name] = {
            pair_id: len(paper_ids)
            for pair_id, paper_ids in papers_by_pair.items()
        }
    return counts


def fitting_subset(
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Sequence[Mapping[str, Any]]],
    pair_paper_counts: Mapping[str, int],
    dimension: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one dimension's examples whose pair has multiple papers."""

    rows = metadata[dimension]
    mask = np.asarray(
        [
            pair_paper_counts.get(str(row["pair_id"]), 0)
            >= MIN_PAIR_PAPER_COUNT
            for row in rows
        ],
        dtype=bool,
    )
    if not np.any(mask):
        raise ValueError(
            f"No {dimension} examples have at least "
            f"{MIN_PAIR_PAPER_COUNT} papers."
        )
    return (
        arrays[f"{dimension}_features"][mask],
        arrays[f"{dimension}_targets"][mask],
        arrays[f"{dimension}_weights"][mask],
    )


def _fit_final_model(
    alpha: float,
    train_subset: tuple[np.ndarray, np.ndarray, np.ndarray],
    validation_subset: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> Pipeline:
    return base.fit_weighted(
        base.make_pipeline(alpha),
        np.vstack([train_subset[0], validation_subset[0]]),
        np.concatenate([train_subset[1], validation_subset[1]]),
        np.concatenate([train_subset[2], validation_subset[2]]),
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=base.DEFAULT_SPLIT_DIR,
        help=f"seven-dimension split directory (default: {base.DEFAULT_SPLIT_DIR})",
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
        if tuple(split_manifest.get("dimensions", ())) != base.DIMENSIONS:
            raise ValueError("Dimension split manifest has unexpected dimensions.")
        if split_manifest.get("integrity", {}).get("pair_overlap_count") != 0:
            raise ValueError("Dimension split manifest reports ORPHA-pair leakage.")

        evaluator = DiseaseDistanceEvaluator(feature_dir=feature_dir)
        arrays, metadata = base.load_dimension_datasets(split_dir, evaluator)
        pair_paper_counts = load_pair_paper_counts(split_dir)
        selections = base.dimension_feature_indices()

        fitting_data: Dict[
            str,
            Dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
        ] = {
            split_name: {
                dimension: fitting_subset(
                    arrays[split_name],
                    metadata[split_name],
                    pair_paper_counts[split_name],
                    dimension,
                )
                for dimension in base.DIMENSIONS
            }
            for split_name in ("train", "validation")
        }

        selected_alphas: Dict[str, float] = {}
        validation_models: Dict[str, Pipeline] = {}
        trials: Dict[str, list[Dict[str, Any]]] = {}
        for dimension in base.DIMENSIONS:
            train = fitting_data["train"][dimension]
            (
                selected_alphas[dimension],
                validation_models[dimension],
                trials[dimension],
            ) = base.select_regularization(
                train[0],
                train[1],
                train[2],
                arrays["validation"][f"{dimension}_features"],
                arrays["validation"][f"{dimension}_targets"],
                arrays["validation"][f"{dimension}_weights"],
            )

        validation_evaluations = evaluate_dimension_regressors(
            validation_models,
            base._dimension_maps(arrays["validation"], "features"),
            base._dimension_maps(arrays["validation"], "targets"),
            base._dimension_maps(arrays["validation"], "weights"),
        )

        final_models = {
            dimension: _fit_final_model(
                selected_alphas[dimension],
                fitting_data["train"][dimension],
                fitting_data["validation"][dimension],
            )
            for dimension in base.DIMENSIONS
        }
        test_evaluations = evaluate_dimension_regressors(
            final_models,
            base._dimension_maps(arrays["test"], "features"),
            base._dimension_maps(arrays["test"], "targets"),
            base._dimension_maps(arrays["test"], "weights"),
        )

        validation_baselines: Dict[str, Dict[str, Any]] = {}
        test_baselines: Dict[str, Dict[str, Any]] = {}
        for dimension in base.DIMENSIONS:
            train = fitting_data["train"][dimension]
            validation = fitting_data["validation"][dimension]
            train_mean = float(np.average(train[1], weights=train[2]))
            final_targets = np.concatenate([train[1], validation[1]])
            final_weights = np.concatenate([train[2], validation[2]])
            final_mean = float(np.average(final_targets, weights=final_weights))
            validation_targets = arrays["validation"][f"{dimension}_targets"]
            validation_weights = arrays["validation"][f"{dimension}_weights"]
            test_targets = arrays["test"][f"{dimension}_targets"]
            test_weights = arrays["test"][f"{dimension}_weights"]
            validation_baselines[dimension] = weighted_regression_metrics(
                validation_targets,
                np.full_like(validation_targets, train_mean),
                validation_weights,
            )
            test_baselines[dimension] = weighted_regression_metrics(
                test_targets,
                np.full_like(test_targets, final_mean),
                test_weights,
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        coefficient_rows, raw_intercepts = base._coefficient_rows(
            final_models,
            selections,
        )
        base._write_csv_atomic(
            output_dir / base.COEFFICIENTS_FILENAME,
            [
                "dimension",
                "feature",
                "standardized_coefficient",
                "raw_unit_coefficient",
            ],
            coefficient_rows,
        )
        prediction_rows = [
            *base._prediction_rows(
                metadata["validation"],
                validation_evaluations,
            ),
            *base._prediction_rows(metadata["test"], test_evaluations),
        ]
        base._write_csv_atomic(
            output_dir / base.PREDICTIONS_FILENAME,
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
                    base.FEATURE_NAMES[index]
                    for index in selections[dimension]
                ]
                for dimension in base.DIMENSIONS
            },
            "selected_alphas": selected_alphas,
            "raw_unit_intercepts": raw_intercepts,
            "dimensions": base.DIMENSIONS,
            "minimum_pair_paper_count": MIN_PAIR_PAPER_COUNT,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "split_manifest_sha256": base._sha256(manifest_path),
        }
        base._dump_model_atomic(output_dir / base.MODEL_FILENAME, artifact)

        dimension_metrics: Dict[str, Any] = {}
        for dimension in base.DIMENSIONS:
            dimension_metrics[dimension] = {
                "selected_alpha": selected_alphas[dimension],
                "feature_count": int(len(selections[dimension])),
                "fitting_counts": {
                    split_name: int(
                        len(fitting_data[split_name][dimension][1])
                    )
                    for split_name in ("train", "validation")
                },
                "evaluation_counts": {
                    split_name: int(
                        len(arrays[split_name][f"{dimension}_targets"])
                    )
                    for split_name in ("validation", "test")
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
            for dimension in base.DIMENSIONS
        ]
        metrics = {
            "schema_version": "1.0.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": {
                "type": "seven_confidence_weighted_ridge_regressions",
                "target": (
                    "dimension-specific confidence-weighted mean score per "
                    "disease pair"
                ),
                "dimensions": list(base.DIMENSIONS),
                "full_pair_feature_count": len(base.FEATURE_NAMES),
                "fitting_filter": {
                    "minimum_distinct_papers_per_pair": MIN_PAIR_PAPER_COUNT,
                    "paper_count_scope": "all dimensions within the pair split",
                },
            },
            "data": {
                "split_directory": str(split_dir),
                "feature_directory": str(feature_dir),
                "split_manifest_sha256": base._sha256(manifest_path),
            },
            "dimensions": dimension_metrics,
            "summary": {
                "macro_confidence_weighted_test_rmse": float(
                    np.mean(weighted_test_rmses)
                )
            },
            "notes": [
                "Only disease pairs supported by at least two distinct papers "
                "were used for fitting.",
                "Validation and test evaluation include all scored pairs.",
                "Validation selected regularization separately for each dimension.",
                "Test data was used once after filtered train+validation refitting.",
                "This model is experimental and is not a clinical tool.",
            ],
        }
        base._write_json_atomic(output_dir / base.METRICS_FILENAME, metrics)
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

    print(f"Trained multi-paper seven-dimension models in {output_dir}")
    for dimension in base.DIMENSIONS:
        weighted = test_evaluations[dimension].metrics["confidence_weighted"]
        baseline = test_baselines[dimension]
        fitted = sum(
            len(fitting_data[split_name][dimension][1])
            for split_name in ("train", "validation")
        )
        print(
            f"  {dimension}: fitted={fitted}, test n={weighted['examples']}, "
            f"RMSE={weighted['root_mean_squared_error']:.3f} "
            f"(baseline {baseline['root_mean_squared_error']:.3f}), "
            f"R2={weighted['r2']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
