"""Test semantic features against their corresponding literature dimensions.

This follow-up isolates label-feature alignment from model complexity.  For the
four dimensions with usable held-out sample sizes, it compares the production
ridge representation with compact semantic features using the same weighted
targets, alpha grid, split, and ridge estimator.  It also reports a sensitivity
test that removes test pairs sharing any source paper with train or validation.

All generated files are constrained to the root ``experiments`` directory.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

sys.dont_write_bytecode = True

import joblib
import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SEMANTIC_SCRIPT = EXPERIMENT_ROOT / "02_semantic_similarity_benchmark.py"
SEMANTIC_SPEC = importlib.util.spec_from_file_location(
    "semantic_similarity_benchmark",
    SEMANTIC_SCRIPT,
)
if SEMANTIC_SPEC is None or SEMANTIC_SPEC.loader is None:
    raise RuntimeError(f"Cannot load semantic helpers from {SEMANTIC_SCRIPT}.")
SEMANTIC = importlib.util.module_from_spec(SEMANTIC_SPEC)
sys.modules[SEMANTIC_SPEC.name] = SEMANTIC
SEMANTIC_SPEC.loader.exec_module(SEMANTIC)

from evaluation import DiseaseDistanceEvaluator  # noqa: E402
from models.evaluation import weighted_regression_metrics  # noqa: E402
from models.linear_regression_7_dimension import (  # noqa: E402
    DEFAULT_ALPHA_GRID,
    dimension_feature_indices,
    fit_weighted,
    load_dimension_datasets,
    make_pipeline,
    select_regularization,
)


DEFAULT_FEATURE_DIR = PROJECT_ROOT / ".data" / "features"
DEFAULT_SPLIT_DIR = PROJECT_ROOT / ".data" / "splits" / "dimensions"
DEFAULT_HPO_OBO = PROJECT_ROOT / ".data" / "databases" / "hpo" / "hp.obo"
DEFAULT_OUTPUT_DIR = EXPERIMENT_ROOT / "artifacts" / "dimension_aligned_v1"
SPLIT_NAMES = ("train", "validation", "test")
PRIMARY_DIMENSIONS = (
    "phenotype",
    "genetic",
    "mechanism",
    "diagnostic_confusability",
)
DIMENSION_PREFIXES = {
    "phenotype": (
        "phenotypes_",
        "body_systems_",
        "onset_",
        "disorder_type_",
        "disorder_group_",
    ),
    "genetic": (
        "genes_",
        "inheritance_",
        "disorder_type_",
        "disorder_group_",
    ),
    "mechanism": (
        "genes_",
        "categories_",
        "ontology_parents_",
        "preferential_parents_",
        "body_systems_",
        "disorder_type_",
        "disorder_group_",
    ),
    "diagnostic_confusability": (
        "phenotypes_",
        "categories_",
        "ontology_parents_",
        "preferential_parents_",
        "body_systems_",
        "onset_",
        "disorder_type_",
        "disorder_group_",
    ),
}


def _required_orpha_ids(split_dir: Path) -> set[str]:
    result: set[str] = set()
    for split_name in SPLIT_NAMES:
        with (split_dir / f"{split_name}.csv").open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as stream:
            for row in csv.DictReader(stream):
                result.update(str(row["_pair_id"]).split("|"))
    return result


def aligned_feature_indices(dimension: str) -> np.ndarray:
    """Select compact semantic columns that directly support a dimension."""

    prefixes = DIMENSION_PREFIXES[dimension]
    names = SEMANTIC.improved_feature_names()
    indices = [
        index
        for index, name in enumerate(names)
        if name.startswith(prefixes)
        and not name.startswith("description_")
    ]
    if not indices:
        raise ValueError(f"No aligned features selected for {dimension}.")
    return np.asarray(indices, dtype=np.int64)


def _build_aligned_matrices(
    metadata: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    context: Any,
) -> Dict[str, Dict[str, np.ndarray]]:
    cache: Dict[str, np.ndarray] = {}
    matrices: Dict[str, Dict[str, np.ndarray]] = {
        split_name: {}
        for split_name in SPLIT_NAMES
    }
    for split_name in SPLIT_NAMES:
        for dimension in PRIMARY_DIMENSIONS:
            indices = aligned_feature_indices(dimension)
            rows: list[np.ndarray] = []
            for item in metadata[split_name][dimension]:
                pair_id = str(item["pair_id"])
                full = cache.get(pair_id)
                if full is None:
                    full = SEMANTIC.extract_improved_features(
                        context,
                        str(item["orpha_id_a"]),
                        str(item["orpha_id_b"]),
                    )
                    cache[pair_id] = full
                rows.append(full[indices])
            matrices[split_name][dimension] = np.vstack(rows)
    return matrices


def _paper_tokens(item: Mapping[str, Any]) -> set[str]:
    return {
        value
        for value in str(item.get("paper_ids") or "").split(";")
        if value
    }


def _fit_final_ridge(
    alpha: float,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_w: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    validation_w: np.ndarray,
) -> Any:
    return fit_weighted(
        make_pipeline(alpha),
        np.vstack([train_x, validation_x]),
        np.concatenate([train_y, validation_y]),
        np.concatenate([train_w, validation_w]),
    )


def _weighted_rmse(
    truth: np.ndarray,
    predictions: np.ndarray,
    weights: np.ndarray,
) -> float:
    return math.sqrt(float(np.average((truth - predictions) ** 2, weights=weights)))


def _paired_bootstrap(
    truth: np.ndarray,
    current_predictions: np.ndarray,
    aligned_predictions: np.ndarray,
    weights: np.ndarray,
    *,
    repetitions: int = 10000,
) -> Dict[str, Any]:
    """Pair bootstrap weighted RMSE deltas; positive favors aligned features."""

    rng = np.random.default_rng(20261004)
    deltas = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        indices = rng.integers(0, len(truth), size=len(truth))
        deltas[repetition] = _weighted_rmse(
            truth[indices],
            current_predictions[indices],
            weights[indices],
        ) - _weighted_rmse(
            truth[indices],
            aligned_predictions[indices],
            weights[indices],
        )
    observed = _weighted_rmse(
        truth,
        current_predictions,
        weights,
    ) - _weighted_rmse(
        truth,
        aligned_predictions,
        weights,
    )
    return {
        "interpretation": "Positive RMSE deltas favor aligned semantic features.",
        "repetitions": repetitions,
        "examples": int(len(truth)),
        "rmse_delta": observed,
        "rmse_delta_95_percentile_interval": np.quantile(
            deltas,
            [0.025, 0.975],
        ).tolist(),
        "rmse_probability_improved": float(np.mean(deltas > 0)),
    }


def _metric_rows(
    dimension: str,
    evaluation_name: str,
    model_metrics: Mapping[str, Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for model_name, metrics in model_metrics.items():
        rows.append(
            {
                "dimension": dimension,
                "evaluation": evaluation_name,
                "model": model_name,
                **metrics,
            }
        )
    return rows


def benchmark_dimensions(
    arrays: Mapping[str, Mapping[str, np.ndarray]],
    metadata: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    aligned: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[Dict[str, Any], list[Dict[str, Any]], list[Dict[str, Any]], Dict[str, Any]]:
    """Compare current and aligned ridge features on the untouched test split."""

    results: Dict[str, Any] = {}
    comparison_rows: list[Dict[str, Any]] = []
    prediction_rows: list[Dict[str, Any]] = []
    models: Dict[str, Any] = {}
    current_selections = dimension_feature_indices()

    for dimension in PRIMARY_DIMENSIONS:
        train_y = arrays["train"][f"{dimension}_targets"]
        validation_y = arrays["validation"][f"{dimension}_targets"]
        test_y = arrays["test"][f"{dimension}_targets"]
        train_w = arrays["train"][f"{dimension}_weights"]
        validation_w = arrays["validation"][f"{dimension}_weights"]
        test_w = arrays["test"][f"{dimension}_weights"]
        current_x = {
            split_name: arrays[split_name][f"{dimension}_features"]
            for split_name in SPLIT_NAMES
        }
        aligned_x = {
            split_name: aligned[split_name][dimension]
            for split_name in SPLIT_NAMES
        }

        current_alpha, _, current_trials = select_regularization(
            current_x["train"],
            train_y,
            train_w,
            current_x["validation"],
            validation_y,
            validation_w,
        )
        aligned_alpha, _, aligned_trials = select_regularization(
            aligned_x["train"],
            train_y,
            train_w,
            aligned_x["validation"],
            validation_y,
            validation_w,
        )
        current_model = _fit_final_ridge(
            current_alpha,
            current_x["train"],
            train_y,
            train_w,
            current_x["validation"],
            validation_y,
            validation_w,
        )
        aligned_model = _fit_final_ridge(
            aligned_alpha,
            aligned_x["train"],
            train_y,
            train_w,
            aligned_x["validation"],
            validation_y,
            validation_w,
        )
        current_predictions = np.asarray(
            current_model.predict(current_x["test"]),
            dtype=np.float64,
        )
        aligned_predictions = np.asarray(
            aligned_model.predict(aligned_x["test"]),
            dtype=np.float64,
        )
        final_y = np.concatenate([train_y, validation_y])
        final_w = np.concatenate([train_w, validation_w])
        final_mean = float(np.average(final_y, weights=final_w))
        mean_predictions = np.full_like(test_y, final_mean)

        test_metrics = {
            "mean_baseline": weighted_regression_metrics(
                test_y,
                mean_predictions,
                test_w,
            ),
            "current_ridge": weighted_regression_metrics(
                test_y,
                current_predictions,
                test_w,
            ),
            "aligned_semantic_ridge": weighted_regression_metrics(
                test_y,
                aligned_predictions,
                test_w,
            ),
        }
        comparison_rows.extend(_metric_rows(dimension, "full_test", test_metrics))

        training_papers: set[str] = set()
        for split_name in ("train", "validation"):
            for item in metadata[split_name][dimension]:
                training_papers.update(_paper_tokens(item))
        blocked_mask = np.asarray(
            [
                not bool(_paper_tokens(item) & training_papers)
                for item in metadata["test"][dimension]
            ],
            dtype=bool,
        )
        blocked_metrics: Optional[Dict[str, Dict[str, Any]]] = None
        if int(np.sum(blocked_mask)) >= 2:
            blocked_metrics = {
                "mean_baseline": weighted_regression_metrics(
                    test_y[blocked_mask],
                    mean_predictions[blocked_mask],
                    test_w[blocked_mask],
                ),
                "current_ridge": weighted_regression_metrics(
                    test_y[blocked_mask],
                    current_predictions[blocked_mask],
                    test_w[blocked_mask],
                ),
                "aligned_semantic_ridge": weighted_regression_metrics(
                    test_y[blocked_mask],
                    aligned_predictions[blocked_mask],
                    test_w[blocked_mask],
                ),
            }
            comparison_rows.extend(
                _metric_rows(dimension, "paper_blocked_test", blocked_metrics)
            )

        bootstrap = _paired_bootstrap(
            test_y,
            current_predictions,
            aligned_predictions,
            test_w,
        )
        aligned_indices = aligned_feature_indices(dimension)
        aligned_names = [
            SEMANTIC.improved_feature_names()[index]
            for index in aligned_indices
        ]
        results[dimension] = {
            "counts": {
                split_name: int(
                    len(arrays[split_name][f"{dimension}_targets"])
                )
                for split_name in SPLIT_NAMES
            },
            "current_feature_count": int(len(current_selections[dimension])),
            "aligned_feature_count": int(len(aligned_names)),
            "aligned_feature_names": aligned_names,
            "current_selected_alpha": current_alpha,
            "aligned_selected_alpha": aligned_alpha,
            "test": test_metrics,
            "paper_blocked_test": blocked_metrics,
            "paper_blocked_removed_pairs": int(np.sum(~blocked_mask)),
            "paired_bootstrap": bootstrap,
            "validation_trials": {
                "current_ridge": current_trials,
                "aligned_semantic_ridge": aligned_trials,
            },
        }
        models[dimension] = {
            "current_ridge": current_model,
            "aligned_semantic_ridge": aligned_model,
            "aligned_feature_names": aligned_names,
        }

        for index, item in enumerate(metadata["test"][dimension]):
            prediction_rows.append(
                {
                    "dimension": dimension,
                    "pair_id": item["pair_id"],
                    "paper_ids": item["paper_ids"],
                    "paper_count": item["paper_count"],
                    "target": float(test_y[index]),
                    "sample_weight": float(test_w[index]),
                    "included_in_paper_blocked_test": int(blocked_mask[index]),
                    "mean_baseline_prediction": float(mean_predictions[index]),
                    "current_ridge_prediction": float(current_predictions[index]),
                    "aligned_semantic_ridge_prediction": float(
                        aligned_predictions[index]
                    ),
                }
            )
    return results, comparison_rows, prediction_rows, models


def _macro_summary(results: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    current_rmse = [
        float(results[dimension]["test"]["current_ridge"]["root_mean_squared_error"])
        for dimension in PRIMARY_DIMENSIONS
    ]
    aligned_rmse = [
        float(
            results[dimension]["test"]["aligned_semantic_ridge"][
                "root_mean_squared_error"
            ]
        )
        for dimension in PRIMARY_DIMENSIONS
    ]
    current_spearman = [
        float(results[dimension]["test"]["current_ridge"]["spearman_correlation"])
        for dimension in PRIMARY_DIMENSIONS
    ]
    aligned_spearman = [
        float(
            results[dimension]["test"]["aligned_semantic_ridge"][
                "spearman_correlation"
            ]
        )
        for dimension in PRIMARY_DIMENSIONS
    ]
    return {
        "dimensions": list(PRIMARY_DIMENSIONS),
        "current_macro_rmse": float(np.mean(current_rmse)),
        "aligned_macro_rmse": float(np.mean(aligned_rmse)),
        "rmse_delta_current_minus_aligned": float(
            np.mean(current_rmse) - np.mean(aligned_rmse)
        ),
        "current_macro_spearman": float(np.mean(current_spearman)),
        "aligned_macro_spearman": float(np.mean(aligned_spearman)),
        "spearman_delta_aligned_minus_current": float(
            np.mean(aligned_spearman) - np.mean(current_spearman)
        ),
        "dimensions_with_lower_aligned_rmse": int(
            sum(
                aligned_value < current_value
                for aligned_value, current_value in zip(
                    aligned_rmse,
                    current_rmse,
                )
            )
        ),
    }


def write_report(path: Path, metrics: Mapping[str, Any]) -> None:
    summary = metrics["summary"]
    rows = []
    for dimension in PRIMARY_DIMENSIONS:
        result = metrics["dimensions"][dimension]
        current = result["test"]["current_ridge"]
        aligned = result["test"]["aligned_semantic_ridge"]
        bootstrap = result["paired_bootstrap"]
        rows.append(
            "<tr>"
            f"<td>{dimension.replace('_', ' ')}</td>"
            f"<td>{float(current['root_mean_squared_error']):.3f}</td>"
            f"<td>{float(aligned['root_mean_squared_error']):.3f}</td>"
            f"<td>{float(current['spearman_correlation']):.3f}</td>"
            f"<td>{float(aligned['spearman_correlation']):.3f}</td>"
            f"<td>{float(bootstrap['rmse_probability_improved']):.1%}</td>"
            f"<td>{int(result['paper_blocked_removed_pairs'])}</td>"
            "</tr>"
        )
    verdict = (
        "Improved"
        if float(summary["rmse_delta_current_minus_aligned"]) > 0
        else "Did not improve"
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dimension-aligned ablation</title>
<style>
:root{{color-scheme:light dark;font-family:Inter,system-ui,sans-serif}}
body{{margin:0;background:Canvas;color:CanvasText}}
main{{max-width:980px;margin:auto;padding:40px 28px 64px}}
h1{{font-size:28px;margin-bottom:6px}}p{{line-height:1.55;max-width:780px}}
.lede,.note{{color:GrayText}}.verdict{{border-block:1px solid GrayText;
padding:20px 0;margin:28px 0;display:flex;justify-content:space-between;gap:24px}}
.verdict strong{{font-size:24px}}table{{width:100%;border-collapse:collapse;margin-top:24px}}
th,td{{text-align:right;padding:11px 8px;border-bottom:1px solid
color-mix(in srgb,CanvasText 20%,Canvas);font-variant-numeric:tabular-nums}}
th:first-child,td:first-child{{text-align:left}}th{{font-size:12px;color:GrayText}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:20px}}
.stat{{border-top:1px solid GrayText;padding-top:12px}}.stat b{{display:block;font-size:22px}}
</style></head><body><main>
<h1>Dimension-aligned feature ablation</h1>
<p class="lede">Same weighted targets, ridge estimator, alpha grid, and untouched
test pairs as the production seven-dimension model.</p>
<section class="verdict"><div>Macro held-out result<br><strong>{verdict}
macro RMSE</strong></div><div><strong>{float(summary["rmse_delta_current_minus_aligned"]):+.4f}</strong>
<br>current − aligned</div></section>
<div class="grid">
<div class="stat">Current macro RMSE<b>{float(summary["current_macro_rmse"]):.3f}</b></div>
<div class="stat">Aligned macro RMSE<b>{float(summary["aligned_macro_rmse"]):.3f}</b></div>
<div class="stat">Current macro Spearman<b>{float(summary["current_macro_spearman"]):.3f}</b></div>
<div class="stat">Aligned macro Spearman<b>{float(summary["aligned_macro_spearman"]):.3f}</b></div>
</div>
<table><thead><tr><th>Dimension</th><th>Current RMSE</th><th>Aligned RMSE</th>
<th>Current ρ</th><th>Aligned ρ</th><th>P(ΔRMSE&gt;0)</th><th>Paper-overlap removed</th>
</tr></thead><tbody>{''.join(rows)}</tbody></table>
<p class="note">Bootstrap probabilities are descriptive, not multiplicity-adjusted.
The paper-blocked sensitivity removes any test pair sharing a source identifier
with train or validation. Sparse therapeutic, natural-history, and comorbidity
dimensions are excluded from macro claims.</p>
</main></body></html>"""
    path.write_text(document, encoding="utf-8")


def run(
    *,
    feature_dir: Path,
    split_dir: Path,
    hpo_obo: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    evaluator = DiseaseDistanceEvaluator(feature_dir=feature_dir)
    arrays, metadata = load_dimension_datasets(split_dir, evaluator)
    required_ids = _required_orpha_ids(split_dir)
    context, source_manifest = SEMANTIC.build_feature_context(
        feature_dir,
        hpo_obo,
        required_ids,
    )
    aligned = _build_aligned_matrices(metadata, context)
    dimensions, comparison_rows, prediction_rows, models = benchmark_dimensions(
        arrays,
        metadata,
        aligned,
    )
    metrics = {
        "schema_version": "1.0.0",
        "experiment": "dimension_aligned_v1",
        "dimensions": dimensions,
        "summary": _macro_summary(dimensions),
        "source_feature_manifest": source_manifest,
        "method": {
            "target": "dimension-specific confidence-weighted pair consensus",
            "sample_weight": "sum of source dimension weights",
            "model": "standardized ridge regression",
            "alpha_grid": list(DEFAULT_ALPHA_GRID),
            "selection": "validation weighted RMSE, then weighted MAE",
            "test": "single evaluation after train+validation refit",
            "paper_blocked_sensitivity": (
                "Remove test pairs sharing any paper_id with train or validation."
            ),
        },
        "limitations": [
            "Dimension scores remain subjective literature annotations.",
            "Dimension weights are reused as sample weights for production comparability.",
            "Test sets contain only 17-41 pairs per included dimension.",
            "Mechanism remains a proxy based on genes and ontology; no pathway graph is available.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    SEMANTIC._write_json(output_dir / "metrics.json", metrics)
    SEMANTIC._write_csv(
        output_dir / "comparison.csv",
        list(comparison_rows[0]),
        comparison_rows,
    )
    SEMANTIC._write_csv(
        output_dir / "test_predictions.csv",
        list(prediction_rows[0]),
        prediction_rows,
    )
    joblib.dump(
        {
            "schema_version": "1.0.0",
            "models": models,
            "semantic_source_script": str(SEMANTIC_SCRIPT),
        },
        output_dir / "models.joblib",
    )
    write_report(output_dir / "report.html", metrics)
    return metrics


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--hpo-obo", type=Path, default=DEFAULT_HPO_OBO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        output_dir = SEMANTIC._assert_output_is_isolated(args.output_dir)
        metrics = run(
            feature_dir=args.feature_dir.expanduser().resolve(),
            split_dir=args.split_dir.expanduser().resolve(),
            hpo_obo=args.hpo_obo.expanduser().resolve(),
            output_dir=output_dir,
        )
    except (FileNotFoundError, KeyError, RuntimeError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    summary = metrics["summary"]
    print(f"Artifacts: {output_dir}")
    print(f"Current macro RMSE: {summary['current_macro_rmse']:.4f}")
    print(f"Aligned macro RMSE: {summary['aligned_macro_rmse']:.4f}")
    print(
        "Delta (current - aligned): "
        f"{summary['rmse_delta_current_minus_aligned']:+.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
