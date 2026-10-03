"""Learn multimodal disease embeddings and evaluate their pairwise similarity.

Each feature modality is independently converted to a sparse disease-term
matrix, TF-IDF weighted, reduced with Truncated SVD, and L2 normalized.  The
modality vectors are concatenated and normalized into one disease embedding.

The embedding itself is unsupervised and transductive over the known ORPHA
catalog.  A small ridge calibrator learns how modality cosine similarities map
to paper-averaged literature scores using train/validation labels only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, normalize


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import DEFAULT_FEATURE_DIR  # noqa: E402
from models.evaluation import classification_metrics, regression_metrics  # noqa: E402
from models.linear_classifier import load_pair_targets  # noqa: E402


DEFAULT_SPLIT_DIR = PROJECT_ROOT / ".data" / "splits"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".data" / "models" / "svd_embeddings"
DEFAULT_ALPHA_GRID = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
MODALITIES: Dict[str, tuple[str, int]] = {
    "phenotypes": ("hpo_ids", 32),
    "genes": ("gene_symbols", 32),
    "classifications": ("category_ids", 24),
    "body_systems": ("body_system_ids", 8),
    "inheritance": ("inheritance", 8),
    "onset": ("onset", 8),
    "approved_drugs": ("approved_drug_ids", 16),
}
MODEL_FILENAME = "model.joblib"
METRICS_FILENAME = "metrics.json"
PREDICTIONS_FILENAME = "predictions.csv"
EMBEDDINGS_FILENAME = "embeddings.parquet"
MODALITIES_FILENAME = "modality_summary.csv"


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


def load_disease_terms(
    feature_dir: Path,
) -> tuple[list[str], list[str], Dict[str, list[list[str]]]]:
    """Load disease identity and aggregate term sets from diseases.parquet."""

    path = feature_dir / "diseases.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"Feature file not found: {path}. Run generate_features.py first."
        )
    columns = ["orpha_id", "name", *(column for column, _ in MODALITIES.values())]
    rows = pq.read_table(path, columns=columns).to_pylist()
    orpha_ids = [str(row["orpha_id"]) for row in rows]
    if len(orpha_ids) != len(set(orpha_ids)):
        raise ValueError("diseases.parquet contains duplicate ORPHA IDs.")
    names = [str(row["name"]) for row in rows]
    terms = {
        modality: [
            sorted(
                {
                    str(value)
                    for value in row.get(column) or []
                    if value not in (None, "")
                }
            )
            for row in rows
        ]
        for modality, (column, _) in MODALITIES.items()
    }
    return orpha_ids, names, terms


def fit_embeddings(
    modality_terms: Mapping[str, Sequence[Sequence[str]]],
    *,
    random_state: int,
) -> tuple[
    np.ndarray,
    Dict[str, np.ndarray],
    Dict[str, Dict[str, Any]],
    list[Dict[str, Any]],
]:
    """Fit TF-IDF/SVD transforms and return combined disease embeddings."""

    modality_embeddings: Dict[str, np.ndarray] = {}
    transforms: Dict[str, Dict[str, Any]] = {}
    summaries: list[Dict[str, Any]] = []

    for modality, (_, requested_components) in MODALITIES.items():
        encoder = MultiLabelBinarizer(sparse_output=True)
        binary = encoder.fit_transform(modality_terms[modality]).astype(np.float64)
        if binary.shape[1] == 0:
            raise ValueError(f"Modality {modality!r} has no terms.")

        tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True)
        weighted = tfidf.fit_transform(binary)
        available = np.asarray(binary.getnnz(axis=1) > 0)
        max_components = min(weighted.shape[0] - 1, weighted.shape[1] - 1)
        if max_components >= 1:
            component_count = min(requested_components, max_components)
            reducer: Optional[TruncatedSVD] = TruncatedSVD(
                n_components=component_count,
                n_iter=7,
                random_state=random_state,
            )
            embedded = reducer.fit_transform(weighted)
            explained_variance = float(
                np.sum(reducer.explained_variance_ratio_)
            )
        else:
            component_count = weighted.shape[1]
            reducer = None
            embedded = weighted.toarray()
            explained_variance = 1.0

        embedded = normalize(embedded, norm="l2", copy=False).astype(np.float32)
        modality_embeddings[modality] = embedded
        transforms[modality] = {
            "encoder": encoder,
            "tfidf": tfidf,
            "reducer": reducer,
            "available": available,
        }
        summaries.append(
            {
                "modality": modality,
                "source_column": MODALITIES[modality][0],
                "terms": int(binary.shape[1]),
                "nonzero_diseases": int(np.sum(available)),
                "requested_components": requested_components,
                "actual_components": component_count,
                "explained_variance_ratio": explained_variance,
            }
        )

    combined = np.concatenate(
        [modality_embeddings[name] for name in MODALITIES],
        axis=1,
    )
    combined = normalize(combined, norm="l2", copy=False).astype(np.float32)
    return combined, modality_embeddings, transforms, summaries


def build_pair_dataset(
    targets: Sequence[Mapping[str, Any]],
    *,
    id_to_index: Mapping[str, int],
    combined_embeddings: np.ndarray,
    modality_embeddings: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Dict[str, Any]]]:
    """Construct pairwise cosine features and continuous literature targets."""

    pair_features: list[list[float]] = []
    raw_similarities: list[float] = []
    scores: list[float] = []
    metadata: list[Dict[str, Any]] = []

    for target in targets:
        disease_a_id, disease_b_id = _parse_pair_id(str(target["pair_id"]))
        try:
            index_a = id_to_index[disease_a_id]
            index_b = id_to_index[disease_b_id]
        except KeyError as error:
            raise KeyError(
                f"{error.args[0]} is absent from the disease embedding cohort."
            ) from error

        overall_cosine = float(
            np.dot(combined_embeddings[index_a], combined_embeddings[index_b])
        )
        values = [overall_cosine]
        modality_scores: Dict[str, float] = {}
        for modality in MODALITIES:
            vector_a = modality_embeddings[modality][index_a]
            vector_b = modality_embeddings[modality][index_b]
            available = bool(np.any(vector_a) and np.any(vector_b))
            cosine = float(np.dot(vector_a, vector_b)) if available else 0.0
            values.extend([cosine, float(available)])
            modality_scores[modality] = cosine

        pair_features.append(values)
        raw_similarities.append(min(1.0, max(0.0, overall_cosine)))
        scores.append(float(target["literature_similarity"]))
        metadata.append(
            {
                **target,
                "orpha_id_a": disease_a_id,
                "orpha_id_b": disease_b_id,
                "overall_cosine": overall_cosine,
                "modality_cosines": modality_scores,
            }
        )

    if not pair_features:
        raise ValueError("A split contains no embedding pairs.")
    return (
        np.asarray(pair_features, dtype=np.float64),
        np.asarray(scores, dtype=np.float64),
        np.asarray(raw_similarities, dtype=np.float64),
        metadata,
    )


def pair_feature_names() -> list[str]:
    return [
        "overall_cosine",
        *(
            name
            for modality in MODALITIES
            for name in (f"{modality}_cosine", f"{modality}_both_available")
        ),
    ]


def make_calibrator(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=alpha)),
        ]
    )


def select_calibrator(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
) -> tuple[float, Pipeline, list[Dict[str, Any]]]:
    """Select ridge calibration strength using validation RMSE."""

    trials: list[Dict[str, Any]] = []
    best_key: Optional[tuple[float, float, float]] = None
    best_alpha = 0.0
    best_model: Optional[Pipeline] = None
    for alpha in DEFAULT_ALPHA_GRID:
        model = make_calibrator(alpha)
        model.fit(train_features, train_targets)
        predictions = model.predict(validation_features)
        metrics = regression_metrics(validation_targets, predictions)
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
        raise RuntimeError("No embedding calibrator was fitted.")
    return best_alpha, best_model, trials


def _prediction_rows(
    split_name: str,
    metadata: Sequence[Mapping[str, Any]],
    raw_similarities: np.ndarray,
    calibrated_predictions: np.ndarray,
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for item, raw_similarity, calibrated in zip(
        metadata,
        raw_similarities,
        calibrated_predictions,
    ):
        target = float(item["literature_similarity"])
        calibrated_value = float(calibrated)
        row: Dict[str, Any] = {
            "split": split_name,
            "pair_id": item["pair_id"],
            "orpha_id_a": item["orpha_id_a"],
            "orpha_id_b": item["orpha_id_b"],
            "literature_similarity": target,
            "literature_similarity_min": item["literature_similarity_min"],
            "literature_similarity_max": item["literature_similarity_max"],
            "literature_observations": item["literature_observations"],
            "relationships": item["relationships"],
            "embedding_similarity": float(raw_similarity),
            "calibrated_similarity": calibrated_value,
            "calibrated_similarity_clipped": min(
                1.0,
                max(0.0, calibrated_value),
            ),
            "absolute_error": abs(calibrated_value - target),
        }
        row.update(
            {
                f"{modality}_cosine": item["modality_cosines"][modality]
                for modality in MODALITIES
            }
        )
        rows.append(row)
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


def _write_embeddings_atomic(
    path: Path,
    orpha_ids: Sequence[str],
    names: Sequence[str],
    embeddings: np.ndarray,
) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        table = pa.table(
            {
                "orpha_id": pa.array(orpha_ids, type=pa.string()),
                "name": pa.array(names, type=pa.string()),
                "embedding": pa.array(
                    embeddings.tolist(),
                    type=pa.list_(pa.float32(), embeddings.shape[1]),
                ),
            }
        )
        pq.write_table(table, temporary, compression="zstd")
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
        help=f"embedding artifact directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="SVD random seed (default: 42)",
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

        orpha_ids, disease_names, modality_terms = load_disease_terms(feature_dir)
        (
            embeddings,
            modality_embeddings,
            transforms,
            modality_summaries,
        ) = fit_embeddings(modality_terms, random_state=args.random_state)
        id_to_index = {
            orpha_id: index for index, orpha_id in enumerate(orpha_ids)
        }

        targets = {
            split_name: load_pair_targets(
                split_dir / f"{split_name}.csv",
                split_name,
            )
            for split_name in ("train", "validation", "test")
        }
        datasets = {
            split_name: build_pair_dataset(
                split_targets,
                id_to_index=id_to_index,
                combined_embeddings=embeddings,
                modality_embeddings=modality_embeddings,
            )
            for split_name, split_targets in targets.items()
        }
        train_x, train_y, train_raw, train_metadata = datasets["train"]
        validation_x, validation_y, validation_raw, validation_metadata = datasets[
            "validation"
        ]
        test_x, test_y, test_raw, test_metadata = datasets["test"]

        selected_alpha, validation_calibrator, trials = select_calibrator(
            train_x,
            train_y,
            validation_x,
            validation_y,
        )
        validation_predictions = validation_calibrator.predict(validation_x)

        final_x = np.vstack([train_x, validation_x])
        final_y = np.concatenate([train_y, validation_y])
        final_calibrator = make_calibrator(selected_alpha)
        final_calibrator.fit(final_x, final_y)
        test_predictions = final_calibrator.predict(test_x)

        validation_metrics = regression_metrics(
            validation_y,
            validation_predictions,
        )
        test_metrics = regression_metrics(test_y, test_predictions)
        raw_validation_metrics = regression_metrics(validation_y, validation_raw)
        raw_test_metrics = regression_metrics(test_y, test_raw)
        validation_baseline = regression_metrics(
            validation_y,
            np.full_like(validation_y, np.mean(train_y)),
        )
        test_baseline = regression_metrics(
            test_y,
            np.full_like(test_y, np.mean(final_y)),
        )
        validation_labels = (validation_y >= 0.5).astype(np.int64)
        test_labels = (test_y >= 0.5).astype(np.int64)
        validation_binary = {
            "raw_embedding_similarity": classification_metrics(
                validation_labels,
                (validation_raw >= 0.5).astype(np.int64),
                positive_scores=validation_raw,
            ),
            "calibrated_similarity": classification_metrics(
                validation_labels,
                (validation_predictions >= 0.5).astype(np.int64),
                positive_scores=validation_predictions,
            ),
        }
        test_binary = {
            "raw_embedding_similarity": classification_metrics(
                test_labels,
                (test_raw >= 0.5).astype(np.int64),
                positive_scores=test_raw,
            ),
            "calibrated_similarity": classification_metrics(
                test_labels,
                (test_predictions >= 0.5).astype(np.int64),
                positive_scores=test_predictions,
            ),
        }

        output_dir.mkdir(parents=True, exist_ok=True)
        _write_embeddings_atomic(
            output_dir / EMBEDDINGS_FILENAME,
            orpha_ids,
            disease_names,
            embeddings,
        )
        _write_csv_atomic(
            output_dir / MODALITIES_FILENAME,
            [
                "modality",
                "source_column",
                "terms",
                "nonzero_diseases",
                "requested_components",
                "actual_components",
                "explained_variance_ratio",
            ],
            modality_summaries,
        )
        prediction_rows = [
            *_prediction_rows(
                "validation",
                validation_metadata,
                validation_raw,
                validation_predictions,
            ),
            *_prediction_rows(
                "test",
                test_metadata,
                test_raw,
                test_predictions,
            ),
        ]
        _write_csv_atomic(
            output_dir / PREDICTIONS_FILENAME,
            [
                "split",
                "pair_id",
                "orpha_id_a",
                "orpha_id_b",
                "literature_similarity",
                "literature_similarity_min",
                "literature_similarity_max",
                "literature_observations",
                "relationships",
                "embedding_similarity",
                "calibrated_similarity",
                "calibrated_similarity_clipped",
                "absolute_error",
                *(f"{modality}_cosine" for modality in MODALITIES),
            ],
            prediction_rows,
        )

        artifact = {
            "orpha_ids": orpha_ids,
            "disease_names": disease_names,
            "embeddings": embeddings,
            "modality_embeddings": modality_embeddings,
            "transforms": transforms,
            "modalities": MODALITIES,
            "pair_feature_names": pair_feature_names(),
            "calibrator": final_calibrator,
            "selected_alpha": selected_alpha,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "split_manifest_sha256": _sha256(manifest_path),
        }
        _dump_model_atomic(output_dir / MODEL_FILENAME, artifact)

        metrics = {
            "schema_version": "1.0.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": {
                "type": "multimodal_tfidf_truncated_svd",
                "embedding_dimensions": int(embeddings.shape[1]),
                "diseases": len(orpha_ids),
                "selected_calibrator_alpha": selected_alpha,
                "pair_feature_names": pair_feature_names(),
                "transductive": True,
            },
            "modalities": modality_summaries,
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
                "calibrated": validation_metrics,
                "raw_embedding_similarity": raw_validation_metrics,
                "mean_target_baseline": validation_baseline,
                "binary_at_0_5": validation_binary,
            },
            "test": {
                "calibrated": test_metrics,
                "raw_embedding_similarity": raw_test_metrics,
                "mean_target_baseline": test_baseline,
                "binary_at_0_5": test_binary,
            },
            "calibrator_trials": trials,
            "notes": [
                "Embeddings use no literature labels.",
                "The ridge calibrator uses train/validation labels only.",
                "The embedding fit is transductive over all known ORPHA diseases.",
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

    print(f"Built multimodal disease embeddings in {output_dir}")
    print(
        f"  diseases: {len(orpha_ids):,}; dimensions: {embeddings.shape[1]}"
    )
    print(
        f"  pairs: {len(train_metadata):,} train, "
        f"{len(validation_metadata):,} validation, {len(test_metadata):,} test"
    )
    print(f"  selected calibrator alpha: {selected_alpha:g}")
    print(
        "  raw test: Spearman "
        f"{raw_test_metrics['spearman_correlation']:.3f}, "
        "ROC AUC "
        f"{test_binary['raw_embedding_similarity']['roc_auc']:.3f}"
    )
    print(
        "  calibrated test: Spearman "
        f"{test_metrics['spearman_correlation']:.3f}, "
        f"RMSE {test_metrics['root_mean_squared_error']:.3f}, "
        f"R2 {test_metrics['r2']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
