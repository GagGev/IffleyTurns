"""Compare the heuristic disease similarity with literature-review scores.

The literature review contains several observations for some disease pairs and
occasionally lists more than one possible ORPHA mapping for a disease.  This
experiment uses unordered ORPHA-ID pairs as its unit of analysis and averages
duplicate literature scores before comparing them with ``evaluation.py``.

Rows with missing or ambiguous ORPHA mappings are skipped by default.  Pass
``--expand-ambiguous`` to evaluate the Cartesian product of ambiguous mappings.
This is opt-in because treating every candidate mapping as ground truth can
artificially inflate the benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DiseaseDistanceEvaluator,
    normalize_orpha_id,
)


DEFAULT_LITERATURE_CSV = (
    PROJECT_ROOT / "literature_review" / "literature_disease_pairs.csv"
)
DEFAULT_OUTPUT_CSV = (
    PROJECT_ROOT
    / ".data"
    / "experiments"
    / "01_compare_evaluation_literature_review.csv"
)
DEFAULT_SUMMARY_JSON = (
    PROJECT_ROOT
    / ".data"
    / "experiments"
    / "01_compare_evaluation_literature_review_summary.json"
)

Pair = Tuple[str, str]

OUTPUT_COLUMNS = [
    "orpha_id_a",
    "orpha_id_b",
    "disease_a",
    "disease_b",
    "literature_similarity",
    "literature_similarity_min",
    "literature_similarity_max",
    "literature_observations",
    "relationships",
    "similarity_dimensions",
    "pmids",
    "evaluation_similarity",
    "evaluation_distance",
    "absolute_error",
    "status",
    "error",
]


def _parse_orpha_ids(value: str) -> list[str]:
    """Parse a semicolon-separated ORPHA mapping and remove duplicates."""

    if not value or not value.strip():
        return []

    ids: list[str] = []
    seen: set[str] = set()
    for part in value.split(";"):
        normalized = normalize_orpha_id(part)
        if normalized not in seen:
            ids.append(normalized)
            seen.add(normalized)
    return ids


def _parse_similarity_score(value: str) -> float:
    """Parse and validate a literature similarity score in [0, 1]."""

    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"similarity score must be in [0, 1], got {value!r}")
    return score


def _ordered_pair(orpha_id_a: str, orpha_id_b: str) -> Pair:
    """Return a stable unordered-pair key."""

    return tuple(sorted((orpha_id_a, orpha_id_b)))  # type: ignore[return-value]


def load_literature_pairs(
    path: Path,
    *,
    expand_ambiguous: bool = False,
) -> tuple[Dict[Pair, Dict[str, Any]], Counter[str]]:
    """Load and aggregate literature evidence by unordered ORPHA-ID pair."""

    if not path.is_file():
        raise FileNotFoundError(f"Literature review CSV not found: {path}")

    pairs: Dict[Pair, Dict[str, Any]] = {}
    counts: Counter[str] = Counter()

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {
            "disease_a_orpha_id",
            "disease_b_orpha_id",
            "similarity_score",
        }
        missing_columns = required - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                "Literature CSV is missing required column(s): "
                + ", ".join(sorted(missing_columns))
            )

        for row_number, row in enumerate(reader, start=2):
            counts["input_rows"] += 1
            try:
                score = _parse_similarity_score(row["similarity_score"])
            except (TypeError, ValueError):
                counts["invalid_similarity_score"] += 1
                continue

            try:
                ids_a = _parse_orpha_ids(row["disease_a_orpha_id"])
                ids_b = _parse_orpha_ids(row["disease_b_orpha_id"])
            except ValueError:
                counts["invalid_orpha_id"] += 1
                continue

            if not ids_a or not ids_b:
                counts["missing_orpha_mapping"] += 1
                continue
            if (len(ids_a) > 1 or len(ids_b) > 1) and not expand_ambiguous:
                counts["ambiguous_orpha_mapping"] += 1
                continue

            row_added = False
            for orpha_id_a in ids_a:
                for orpha_id_b in ids_b:
                    if orpha_id_a == orpha_id_b:
                        counts["same_disease_pair"] += 1
                        continue

                    pair = _ordered_pair(orpha_id_a, orpha_id_b)
                    evidence = pairs.setdefault(
                        pair,
                        {
                            "scores": [],
                            "relationships": set(),
                            "dimensions": set(),
                            "pmids": set(),
                            "source_rows": set(),
                        },
                    )
                    evidence["scores"].append(score)
                    for relationship in row.get("relationship", "").split(";"):
                        if relationship.strip():
                            evidence["relationships"].add(relationship.strip())
                    for dimension in row.get("similarity_dimension", "").split(";"):
                        if dimension.strip():
                            evidence["dimensions"].add(dimension.strip())
                    if row.get("pmid", "").strip():
                        evidence["pmids"].add(row["pmid"].strip())
                    evidence["source_rows"].add(row_number)
                    row_added = True

            if row_added:
                counts["usable_rows"] += 1

    counts["unique_orpha_pairs"] = len(pairs)
    return pairs, counts


def compare_pairs(
    pairs: Mapping[Pair, Mapping[str, Any]],
    evaluator: DiseaseDistanceEvaluator,
) -> list[Dict[str, Any]]:
    """Calculate evaluation scores for every aggregated literature pair."""

    rows: list[Dict[str, Any]] = []
    for (orpha_id_a, orpha_id_b), evidence in sorted(pairs.items()):
        scores = evidence["scores"]
        row: Dict[str, Any] = {
            "orpha_id_a": orpha_id_a,
            "orpha_id_b": orpha_id_b,
            "disease_a": "",
            "disease_b": "",
            "literature_similarity": statistics.fmean(scores),
            "literature_similarity_min": min(scores),
            "literature_similarity_max": max(scores),
            "literature_observations": len(evidence["source_rows"]),
            "relationships": " | ".join(sorted(evidence["relationships"])),
            "similarity_dimensions": " | ".join(sorted(evidence["dimensions"])),
            "pmids": " | ".join(sorted(evidence["pmids"])),
            "evaluation_similarity": "",
            "evaluation_distance": "",
            "absolute_error": "",
            "status": "unavailable",
            "error": "",
        }

        try:
            result = evaluator.calculate_distance(orpha_id_a, orpha_id_b)
        except (KeyError, ValueError, OSError) as error:
            row["error"] = str(error)
        else:
            evaluation_similarity = float(result["similarity"])
            row.update(
                {
                    "disease_a": result["disease_a"]["name"],
                    "disease_b": result["disease_b"]["name"],
                    "evaluation_similarity": evaluation_similarity,
                    "evaluation_distance": float(result["distance"]),
                    "absolute_error": abs(
                        evaluation_similarity - row["literature_similarity"]
                    ),
                    "status": "evaluated",
                }
            )
        rows.append(row)
    return rows


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Return one-based ranks, assigning tied values their average rank."""

    ranked = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    start = 0
    while start < len(ranked):
        end = start + 1
        while end < len(ranked) and ranked[end][1] == ranked[start][1]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for index, _ in ranked[start:end]:
            result[index] = average_rank
        start = end
    return result


def _pearson(values_a: Sequence[float], values_b: Sequence[float]) -> Optional[float]:
    """Calculate Pearson correlation, or None for insufficient variation."""

    if len(values_a) != len(values_b) or len(values_a) < 2:
        return None
    mean_a = statistics.fmean(values_a)
    mean_b = statistics.fmean(values_b)
    deviations_a = [value - mean_a for value in values_a]
    deviations_b = [value - mean_b for value in values_b]
    denominator = math.sqrt(
        sum(value * value for value in deviations_a)
        * sum(value * value for value in deviations_b)
    )
    if denominator == 0:
        return None
    return sum(
        value_a * value_b
        for value_a, value_b in zip(deviations_a, deviations_b)
    ) / denominator


def calculate_summary(
    rows: Iterable[Mapping[str, Any]],
    load_counts: Mapping[str, int],
) -> Dict[str, Any]:
    """Calculate coverage, error, and correlation metrics."""

    row_list = list(rows)
    evaluated = [row for row in row_list if row["status"] == "evaluated"]
    literature = [float(row["literature_similarity"]) for row in evaluated]
    predicted = [float(row["evaluation_similarity"]) for row in evaluated]
    errors = [
        abs(reference - prediction)
        for reference, prediction in zip(literature, predicted)
    ]

    summary: Dict[str, Any] = {
        "loading": dict(load_counts),
        "pair_coverage": {
            "total_unique_pairs": len(row_list),
            "evaluated_pairs": len(evaluated),
            "unavailable_pairs": len(row_list) - len(evaluated),
            "fraction_evaluated": (
                len(evaluated) / len(row_list) if row_list else 0.0
            ),
        },
        "metrics": {
            "mean_absolute_error": statistics.fmean(errors) if errors else None,
            "root_mean_squared_error": (
                math.sqrt(statistics.fmean(error * error for error in errors))
                if errors
                else None
            ),
            "pearson_correlation": _pearson(literature, predicted),
            "spearman_correlation": _pearson(
                _average_ranks(literature),
                _average_ranks(predicted),
            ),
        },
    }
    return summary


def write_results(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write pair-level comparison results as CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, summary: Mapping[str, Any]) -> None:
    """Write the machine-readable experiment summary."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
        file.write("\n")


def _format_metric(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def format_summary(
    summary: Mapping[str, Any],
    output_path: Path,
    summary_path: Path,
) -> str:
    """Format a concise human-readable experiment summary."""

    loading = summary["loading"]
    coverage = summary["pair_coverage"]
    metrics = summary["metrics"]
    lines = [
        "Literature-review comparison complete",
        (
            f"Rows: {loading.get('input_rows', 0):,} input, "
            f"{loading.get('usable_rows', 0):,} usable"
        ),
        (
            f"Unique ORPHA pairs: {coverage['total_unique_pairs']:,}; "
            f"evaluated: {coverage['evaluated_pairs']:,} "
            f"({coverage['fraction_evaluated']:.1%})"
        ),
        (
            "Skipped mappings: "
            f"{loading.get('missing_orpha_mapping', 0):,} missing, "
            f"{loading.get('ambiguous_orpha_mapping', 0):,} ambiguous, "
            f"{loading.get('invalid_orpha_id', 0):,} invalid"
        ),
        f"Mean absolute error: {_format_metric(metrics['mean_absolute_error'])}",
        f"RMSE: {_format_metric(metrics['root_mean_squared_error'])}",
        f"Pearson correlation: {_format_metric(metrics['pearson_correlation'])}",
        f"Spearman correlation: {_format_metric(metrics['spearman_correlation'])}",
        f"Pair results: {output_path}",
        f"Summary: {summary_path}",
    ]
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--literature-csv",
        type=Path,
        default=DEFAULT_LITERATURE_CSV,
        help=f"literature-review pair CSV (default: {DEFAULT_LITERATURE_CSV})",
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=DEFAULT_FEATURE_DIR,
        help=f"generated feature directory (default: {DEFAULT_FEATURE_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"pair-level output CSV (default: {DEFAULT_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=DEFAULT_SUMMARY_JSON,
        help=f"summary JSON output (default: {DEFAULT_SUMMARY_JSON})",
    )
    parser.add_argument(
        "--expand-ambiguous",
        action="store_true",
        help="evaluate every combination when a disease has multiple ORPHA IDs",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the literature-review comparison experiment."""

    args = parse_args(argv)
    try:
        pairs, load_counts = load_literature_pairs(
            args.literature_csv.expanduser().resolve(),
            expand_ambiguous=args.expand_ambiguous,
        )
        if not pairs:
            raise ValueError("No usable ORPHA-ID pairs were found in the literature CSV.")
        evaluator = DiseaseDistanceEvaluator(feature_dir=args.feature_dir)
        rows = compare_pairs(pairs, evaluator)
        summary = calculate_summary(rows, load_counts)
        output_path = args.output.expanduser().resolve()
        summary_path = args.summary_json.expanduser().resolve()
        write_results(output_path, rows)
        write_summary(summary_path, summary)
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(format_summary(summary, output_path, summary_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
