"""Model-agnostic evaluation utilities.

The functions in this module operate on estimator behaviour rather than a
specific implementation.  A classifier only needs ``predict``; probability
and ranking metrics are added when it also exposes ``predict_proba`` or
``decision_function``.  A regressor only needs ``predict``.

This module deliberately does not load project data or train models.  Callers
remain responsible for constructing leakage-free datasets and can therefore
reuse these evaluators with linear models, trees, ensembles, neural-network
adapters, or other estimators with compatible methods.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Sequence, runtime_checkable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


ArrayLike = Sequence[Any] | np.ndarray


@runtime_checkable
class Predictor(Protocol):
    """Minimal protocol shared by classifiers and regressors."""

    def predict(self, features: Any) -> Any:
        """Return one prediction per input row."""


@dataclass(frozen=True)
class ClassificationEvaluation:
    """Metrics and row-level outputs from a binary classifier."""

    metrics: Dict[str, Any]
    predictions: np.ndarray
    positive_scores: Optional[np.ndarray]
    positive_probabilities: Optional[np.ndarray]


@dataclass(frozen=True)
class RegressionEvaluation:
    """Metrics and row-level outputs from a regressor."""

    metrics: Dict[str, Any]
    predictions: np.ndarray


def _one_dimensional(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}.")
    if len(array) == 0:
        raise ValueError(f"{name} must not be empty.")
    return array


def _estimator_classes(model: Any) -> Optional[np.ndarray]:
    classes = getattr(model, "classes_", None)
    if classes is None:
        return None
    result = np.asarray(classes)
    return result if result.ndim == 1 else None


def _positive_class_index(model: Any, positive_label: Any, width: int) -> int:
    classes = _estimator_classes(model)
    if classes is None:
        if width != 2:
            raise ValueError(
                "A classifier without classes_ must return exactly two score columns."
            )
        return 1

    matches = np.flatnonzero(classes == positive_label)
    if len(matches) != 1:
        raise ValueError(
            f"Positive label {positive_label!r} is absent from model classes "
            f"{classes.tolist()}."
        )
    return int(matches[0])


def classifier_outputs(
    model: Predictor,
    features: Any,
    *,
    positive_label: Any = 1,
    probability_threshold: Optional[float] = None,
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Return binary predictions, ranking scores, and calibrated probabilities.

    ``positive_scores`` are suitable for ROC AUC and average precision.  They
    are probabilities when available and decision-function values otherwise.
    ``positive_probabilities`` is only populated for ``predict_proba`` models,
    because arbitrary decision scores must not be presented as calibrated
    probabilities.
    """

    if probability_threshold is not None and not 0.0 <= probability_threshold <= 1.0:
        raise ValueError("probability_threshold must be in [0, 1].")

    probabilities: Optional[np.ndarray] = None
    scores: Optional[np.ndarray] = None

    predict_proba = getattr(model, "predict_proba", None)
    if callable(predict_proba):
        all_probabilities = np.asarray(predict_proba(features), dtype=np.float64)
        if all_probabilities.ndim != 2:
            raise ValueError(
                "predict_proba must return a two-dimensional array; got "
                f"shape {all_probabilities.shape}."
            )
        positive_index = _positive_class_index(
            model,
            positive_label,
            all_probabilities.shape[1],
        )
        probabilities = all_probabilities[:, positive_index]
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("predict_proba returned non-finite values.")
        if np.any((probabilities < 0.0) | (probabilities > 1.0)):
            raise ValueError("predict_proba returned values outside [0, 1].")
        scores = probabilities

    if scores is None:
        decision_function = getattr(model, "decision_function", None)
        if callable(decision_function):
            decisions = np.asarray(decision_function(features), dtype=np.float64)
            classes = _estimator_classes(model)
            if decisions.ndim == 1:
                scores = decisions
                if (
                    classes is not None
                    and len(classes) == 2
                    and classes[0] == positive_label
                ):
                    scores = -scores
            elif decisions.ndim == 2:
                positive_index = _positive_class_index(
                    model,
                    positive_label,
                    decisions.shape[1],
                )
                scores = decisions[:, positive_index]
            else:
                raise ValueError(
                    "decision_function must return a one- or two-dimensional "
                    f"array; got shape {decisions.shape}."
                )
            if not np.all(np.isfinite(scores)):
                raise ValueError("decision_function returned non-finite values.")

    if probability_threshold is not None:
        if probabilities is None:
            raise ValueError(
                "probability_threshold requires a model with predict_proba."
            )
        predictions = (probabilities >= probability_threshold).astype(np.int64)
    else:
        raw_predictions = _one_dimensional(
            np.asarray(model.predict(features)),
            "model predictions",
        )
        predictions = (raw_predictions == positive_label).astype(np.int64)

    expected_rows = len(predictions)
    if scores is not None and len(scores) != expected_rows:
        raise ValueError("Model score and prediction row counts differ.")
    if probabilities is not None and len(probabilities) != expected_rows:
        raise ValueError("Model probability and prediction row counts differ.")
    return predictions, scores, probabilities


def classification_metrics(
    true_labels: ArrayLike,
    predictions: ArrayLike,
    *,
    positive_scores: Optional[ArrayLike] = None,
    positive_probabilities: Optional[ArrayLike] = None,
    positive_label: Any = 1,
) -> Dict[str, Any]:
    """Calculate binary metrics, omitting metrics unsupported by model output."""

    raw_labels = _one_dimensional(true_labels, "true_labels")
    binary_labels = (raw_labels == positive_label).astype(np.int64)
    binary_predictions = _one_dimensional(predictions, "predictions").astype(
        np.int64
    )
    if len(binary_labels) != len(binary_predictions):
        raise ValueError("true_labels and predictions must have equal lengths.")
    if not np.all(np.isin(binary_predictions, [0, 1])):
        raise ValueError("predictions must contain only binary 0/1 values.")

    scores = (
        _one_dimensional(positive_scores, "positive_scores").astype(np.float64)
        if positive_scores is not None
        else None
    )
    probabilities = (
        _one_dimensional(
            positive_probabilities,
            "positive_probabilities",
        ).astype(np.float64)
        if positive_probabilities is not None
        else None
    )
    if scores is not None and len(scores) != len(binary_labels):
        raise ValueError("positive_scores and true_labels must have equal lengths.")
    if probabilities is not None and len(probabilities) != len(binary_labels):
        raise ValueError(
            "positive_probabilities and true_labels must have equal lengths."
        )

    matrix = confusion_matrix(binary_labels, binary_predictions, labels=[0, 1])
    class_counts = np.bincount(binary_labels, minlength=2)
    has_both_classes = bool(class_counts[0] and class_counts[1])
    metrics: Dict[str, Any] = {
        "examples": int(len(binary_labels)),
        "class_counts": {
            "negative": int(class_counts[0]),
            "positive": int(class_counts[1]),
        },
        "accuracy": float(accuracy_score(binary_labels, binary_predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(binary_labels, binary_predictions)
        ),
        "precision": float(
            precision_score(binary_labels, binary_predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(binary_labels, binary_predictions, zero_division=0)
        ),
        "f1": float(f1_score(binary_labels, binary_predictions, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(binary_labels, scores))
            if has_both_classes and scores is not None
            else None
        ),
        "average_precision": (
            float(average_precision_score(binary_labels, scores))
            if has_both_classes and scores is not None
            else None
        ),
        "log_loss": (
            float(log_loss(binary_labels, probabilities, labels=[0, 1]))
            if probabilities is not None
            else None
        ),
        "confusion_matrix": {
            "true_negative": int(matrix[0, 0]),
            "false_positive": int(matrix[0, 1]),
            "false_negative": int(matrix[1, 0]),
            "true_positive": int(matrix[1, 1]),
        },
    }
    return metrics


def evaluate_classifier(
    model: Predictor,
    features: Any,
    true_labels: ArrayLike,
    *,
    positive_label: Any = 1,
    probability_threshold: Optional[float] = None,
) -> ClassificationEvaluation:
    """Evaluate any compatible binary classifier."""

    predictions, scores, probabilities = classifier_outputs(
        model,
        features,
        positive_label=positive_label,
        probability_threshold=probability_threshold,
    )
    metrics = classification_metrics(
        true_labels,
        predictions,
        positive_scores=scores,
        positive_probabilities=probabilities,
        positive_label=positive_label,
    )
    return ClassificationEvaluation(
        metrics=metrics,
        predictions=predictions,
        positive_scores=scores,
        positive_probabilities=probabilities,
    )


def regression_metrics(
    true_values: ArrayLike,
    predictions: ArrayLike,
) -> Dict[str, Any]:
    """Calculate model-agnostic continuous-target metrics."""

    truth = _one_dimensional(true_values, "true_values").astype(np.float64)
    predicted = _one_dimensional(predictions, "predictions").astype(np.float64)
    if len(truth) != len(predicted):
        raise ValueError("true_values and predictions must have equal lengths.")
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(predicted)):
        raise ValueError("Regression values must all be finite.")

    truth_centered = truth - np.mean(truth)
    predicted_centered = predicted - np.mean(predicted)
    denominator = math.sqrt(
        float(np.dot(truth_centered, truth_centered))
        * float(np.dot(predicted_centered, predicted_centered))
    )
    pearson = (
        float(np.dot(truth_centered, predicted_centered) / denominator)
        if denominator > 0
        else None
    )
    truth_ranks = _average_ranks(truth)
    predicted_ranks = _average_ranks(predicted)
    rank_truth_centered = truth_ranks - np.mean(truth_ranks)
    rank_predicted_centered = predicted_ranks - np.mean(predicted_ranks)
    rank_denominator = math.sqrt(
        float(np.dot(rank_truth_centered, rank_truth_centered))
        * float(np.dot(rank_predicted_centered, rank_predicted_centered))
    )
    spearman = (
        float(
            np.dot(rank_truth_centered, rank_predicted_centered)
            / rank_denominator
        )
        if rank_denominator > 0
        else None
    )
    return {
        "examples": int(len(truth)),
        "mean_absolute_error": float(mean_absolute_error(truth, predicted)),
        "root_mean_squared_error": float(
            math.sqrt(mean_squared_error(truth, predicted))
        ),
        "r2": float(r2_score(truth, predicted)) if len(truth) >= 2 else None,
        "pearson_correlation": pearson,
        "spearman_correlation": spearman,
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks with deterministic tie handling."""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = ((start + 1) + end) / 2.0
        start = end
    return ranks


def evaluate_regressor(
    model: Predictor,
    features: Any,
    true_values: ArrayLike,
) -> RegressionEvaluation:
    """Evaluate any compatible continuous-target regressor."""

    predictions = _one_dimensional(
        np.asarray(model.predict(features), dtype=np.float64),
        "model predictions",
    )
    return RegressionEvaluation(
        metrics=regression_metrics(true_values, predictions),
        predictions=predictions,
    )
