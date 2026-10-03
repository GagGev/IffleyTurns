"""Create stable train, validation, and test splits for literature evidence.

The model input is an unordered pair of ORPHA diseases, while papers provide
the target similarity score.  All observations for the same ORPHA pair are
therefore assigned to the same split.  This prevents the model from seeing an
identical pair during training and evaluation.

Assignment uses a deterministic hash of the pair instead of shuffling rows.
Consequently, appending new papers does not move existing pairs between
splits, and a new paper about an existing pair automatically follows that
pair's previous assignment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

try:
    import pyarrow.parquet as pq
except ImportError:  # Keep --help available without the optional dependency.
    pq = None

from evaluation import normalize_orpha_id


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "literature_review" / "literature_disease_pairs.csv"
DEFAULT_FEATURE_DIR = PROJECT_ROOT / ".data" / "features"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".data" / "splits"
DEFAULT_SALT = "rare-disease-literature-splits-v1"
SPLIT_NAMES = ("train", "validation", "test")
METADATA_COLUMNS = ("_pair_id", "_paper_id", "_split")
REJECTION_COLUMNS = ("_rejection_reason", "_rejection_detail")


def _parse_orpha_ids(value: str) -> list[str]:
    """Parse a semicolon-separated ORPHA mapping into unique normalized IDs."""

    if not value or not value.strip():
        return []
    result: list[str] = []
    seen: set[str] = set()
    for part in value.split(";"):
        normalized = normalize_orpha_id(part)
        if normalized not in seen:
            result.append(normalized)
            seen.add(normalized)
    return result


def _orpha_sort_key(orpha_id: str) -> int:
    return int(orpha_id.split(":", 1)[1])


def make_pair_id(orpha_id_a: str, orpha_id_b: str) -> str:
    """Build a canonical unordered ORPHA-pair identifier."""

    first, second = sorted((orpha_id_a, orpha_id_b), key=_orpha_sort_key)
    return f"{first}|{second}"


def make_paper_id(row: Mapping[str, str]) -> str:
    """Return the most durable available identifier for a source paper."""

    pmid = row.get("pmid", "").strip()
    if pmid:
        return f"PMID:{pmid}"

    link = row.get("link", "").strip()
    if link:
        return f"LINK:{link.casefold()}"

    title = " ".join(row.get("paper_title", "").split())
    if title:
        return f"TITLE:{title.casefold()}"

    # This fallback is content-based, so inserting earlier rows does not change
    # the identifier of an existing source record.
    serialized = json.dumps(dict(row), sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"ROW:{digest}"


def _validate_score(value: str) -> None:
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("similarity_score must be a finite number in [0, 1]")


def assign_split(
    pair_id: str,
    *,
    train_ratio: float,
    validation_ratio: float,
    salt: str,
) -> str:
    """Assign a pair to a stable split using ratio-based hash ranges."""

    digest = hashlib.blake2b(
        f"{salt}\0{pair_id}".encode("utf-8"),
        digest_size=8,
    ).digest()
    fraction = int.from_bytes(digest, "big") / 2**64
    if fraction < train_ratio:
        return "train"
    if fraction < train_ratio + validation_ratio:
        return "validation"
    return "test"


def load_feature_orpha_ids(feature_dir: Path) -> set[str]:
    """Load the ORPHA cohort for which generated model features exist."""

    if pq is None:
        raise RuntimeError(
            "pyarrow is required to check generated features. Install it with "
            "`python -m pip install pyarrow`, or use "
            "`--include-pairs-without-features`."
        )

    disease_path = feature_dir / "diseases.parquet"
    if not disease_path.is_file():
        raise FileNotFoundError(
            f"Feature file not found: {disease_path}. Run generate_features.py "
            "first, or use --include-pairs-without-features."
        )
    table = pq.read_table(disease_path, columns=["orpha_id"])
    return {
        str(orpha_id)
        for orpha_id in table.column("orpha_id").to_pylist()
        if orpha_id
    }


def _reject(
    row: Mapping[str, str],
    *,
    fieldnames: Sequence[str],
    paper_id: str,
    reason: str,
    detail: str = "",
    pair_id: str = "",
) -> Dict[str, str]:
    rejected = {field: row.get(field, "") for field in fieldnames}
    rejected.update(
        {
            "_pair_id": pair_id,
            "_paper_id": paper_id,
            "_split": "",
            "_rejection_reason": reason,
            "_rejection_detail": detail,
        }
    )
    return rejected


def split_rows(
    csv_bytes: bytes,
    *,
    feature_orpha_ids: Optional[set[str]],
    train_ratio: float,
    validation_ratio: float,
    salt: str,
    expand_ambiguous: bool,
) -> tuple[
    Dict[str, list[Dict[str, str]]],
    list[Dict[str, str]],
    list[str],
    Counter[str],
]:
    """Validate, enrich, and split one immutable snapshot of the source CSV."""

    text = csv_bytes.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    fieldnames = list(reader.fieldnames or [])
    required = {
        "disease_a_orpha_id",
        "disease_b_orpha_id",
        "similarity_score",
    }
    missing = required - set(fieldnames)
    if missing:
        raise ValueError(
            "Input CSV is missing required column(s): " + ", ".join(sorted(missing))
        )
    conflicting = (set(METADATA_COLUMNS) | set(REJECTION_COLUMNS)) & set(fieldnames)
    if conflicting:
        raise ValueError(
            "Input CSV already contains reserved output column(s): "
            + ", ".join(sorted(conflicting))
        )

    splits: Dict[str, list[Dict[str, str]]] = {
        split_name: [] for split_name in SPLIT_NAMES
    }
    rejected: list[Dict[str, str]] = []
    counts: Counter[str] = Counter()

    for source_row in reader:
        counts["input_rows"] += 1
        row = {field: source_row.get(field, "") or "" for field in fieldnames}
        paper_id = make_paper_id(row)

        try:
            _validate_score(row["similarity_score"])
        except (TypeError, ValueError):
            counts["rejected_invalid_similarity_score"] += 1
            rejected.append(
                _reject(
                    row,
                    fieldnames=fieldnames,
                    paper_id=paper_id,
                    reason="invalid_similarity_score",
                    detail=row["similarity_score"],
                )
            )
            continue

        try:
            ids_a = _parse_orpha_ids(row["disease_a_orpha_id"])
            ids_b = _parse_orpha_ids(row["disease_b_orpha_id"])
        except ValueError as error:
            counts["rejected_invalid_orpha_id"] += 1
            rejected.append(
                _reject(
                    row,
                    fieldnames=fieldnames,
                    paper_id=paper_id,
                    reason="invalid_orpha_id",
                    detail=str(error),
                )
            )
            continue

        if not ids_a or not ids_b:
            missing_sides = (
                "both"
                if not ids_a and not ids_b
                else "disease_a"
                if not ids_a
                else "disease_b"
            )
            counts["rejected_missing_orpha_mapping"] += 1
            rejected.append(
                _reject(
                    row,
                    fieldnames=fieldnames,
                    paper_id=paper_id,
                    reason="missing_orpha_mapping",
                    detail=missing_sides,
                )
            )
            continue

        if (len(ids_a) > 1 or len(ids_b) > 1) and not expand_ambiguous:
            counts["rejected_ambiguous_orpha_mapping"] += 1
            rejected.append(
                _reject(
                    row,
                    fieldnames=fieldnames,
                    paper_id=paper_id,
                    reason="ambiguous_orpha_mapping",
                    detail=(
                        f"{row['disease_a_orpha_id']} | "
                        f"{row['disease_b_orpha_id']}"
                    ),
                )
            )
            continue

        row_was_eligible = False
        for orpha_id_a in ids_a:
            for orpha_id_b in ids_b:
                pair_id = make_pair_id(orpha_id_a, orpha_id_b)
                if orpha_id_a == orpha_id_b:
                    counts["rejected_same_disease_pair"] += 1
                    rejected.append(
                        _reject(
                            row,
                            fieldnames=fieldnames,
                            paper_id=paper_id,
                            reason="same_disease_pair",
                            pair_id=pair_id,
                        )
                    )
                    continue

                if feature_orpha_ids is not None:
                    unavailable = sorted(
                        {orpha_id_a, orpha_id_b} - feature_orpha_ids,
                        key=_orpha_sort_key,
                    )
                    if unavailable:
                        counts["rejected_feature_not_generated"] += 1
                        rejected.append(
                            _reject(
                                row,
                                fieldnames=fieldnames,
                                paper_id=paper_id,
                                reason="feature_not_generated",
                                detail=";".join(unavailable),
                                pair_id=pair_id,
                            )
                        )
                        continue

                split_name = assign_split(
                    pair_id,
                    train_ratio=train_ratio,
                    validation_ratio=validation_ratio,
                    salt=salt,
                )
                enriched = dict(row)
                enriched.update(
                    {
                        "_pair_id": pair_id,
                        "_paper_id": paper_id,
                        "_split": split_name,
                    }
                )
                splits[split_name].append(enriched)
                counts[f"{split_name}_rows"] += 1
                row_was_eligible = True

        if row_was_eligible:
            counts["eligible_source_rows"] += 1

    counts["eligible_observations"] = sum(len(rows) for rows in splits.values())
    counts["rejected_observations"] = len(rejected)
    return splits, rejected, fieldnames, counts


def _write_csv_atomic(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    """Replace a CSV only after its complete successor has been written."""

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


def _build_manifest(
    *,
    input_path: Path,
    input_hash: str,
    feature_dir: Path,
    feature_filter_enabled: bool,
    output_dir: Path,
    splits: Mapping[str, Sequence[Mapping[str, str]]],
    rejected: Sequence[Mapping[str, str]],
    counts: Mapping[str, int],
    train_ratio: float,
    validation_ratio: float,
    salt: str,
    expand_ambiguous: bool,
) -> Dict[str, Any]:
    pair_splits: Dict[str, set[str]] = defaultdict(set)
    paper_splits: Dict[str, set[str]] = defaultdict(set)
    split_statistics: Dict[str, Any] = {}

    for split_name, rows in splits.items():
        pairs = {row["_pair_id"] for row in rows}
        papers = {row["_paper_id"] for row in rows}
        split_statistics[split_name] = {
            "rows": len(rows),
            "unique_pairs": len(pairs),
            "unique_papers": len(papers),
        }
        for pair_id in pairs:
            pair_splits[pair_id].add(split_name)
        for paper_id in papers:
            paper_splits[paper_id].add(split_name)

    rejection_reasons = Counter(
        row["_rejection_reason"] for row in rejected
    )
    return {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "file": str(input_path),
            "sha256": input_hash,
            "rows": counts.get("input_rows", 0),
        },
        "features": {
            "filter_enabled": feature_filter_enabled,
            "directory": str(feature_dir),
        },
        "output_directory": str(output_dir),
        "strategy": {
            "group_by": "unordered_orpha_pair",
            "assignment": "blake2b_hash_range",
            "salt": salt,
            "train_ratio": train_ratio,
            "validation_ratio": validation_ratio,
            "test_ratio": round(
                max(0.0, 1.0 - train_ratio - validation_ratio),
                12,
            ),
            "expand_ambiguous_mappings": expand_ambiguous,
            "update_behavior": (
                "Existing ORPHA pairs retain their split when source rows are "
                "added; new evidence for a known pair follows that pair."
            ),
        },
        "splits": split_statistics,
        "integrity": {
            "pair_overlap_count": sum(
                len(split_names) > 1 for split_names in pair_splits.values()
            ),
            "papers_spanning_multiple_splits": sum(
                len(split_names) > 1 for split_names in paper_splits.values()
            ),
        },
        "counts": dict(counts),
        "rejections_by_reason": dict(sorted(rejection_reasons.items())),
    }


def write_outputs(
    output_dir: Path,
    *,
    splits: Mapping[str, Sequence[Mapping[str, str]]],
    rejected: Sequence[Mapping[str, str]],
    source_fieldnames: Sequence[str],
    manifest: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_fieldnames = [*source_fieldnames, *METADATA_COLUMNS]
    rejected_fieldnames = [
        *source_fieldnames,
        *METADATA_COLUMNS,
        *REJECTION_COLUMNS,
    ]
    for split_name in SPLIT_NAMES:
        _write_csv_atomic(
            output_dir / f"{split_name}.csv",
            split_fieldnames,
            splits[split_name],
        )
    _write_csv_atomic(output_dir / "rejected.csv", rejected_fieldnames, rejected)
    _write_json_atomic(output_dir / "split_manifest.json", manifest)


def _ratio(value: str) -> float:
    try:
        ratio = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("ratio must be a number") from error
    if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise argparse.ArgumentTypeError("ratio must be in [0, 1]")
    return ratio


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"literature evidence CSV (default: {DEFAULT_INPUT})",
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
        help=f"split output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--train-ratio",
        type=_ratio,
        default=0.8,
        help="stable hash range allocated to training (default: 0.8)",
    )
    parser.add_argument(
        "--validation-ratio",
        type=_ratio,
        default=0.1,
        help="stable hash range allocated to validation (default: 0.1)",
    )
    parser.add_argument(
        "--salt",
        default=DEFAULT_SALT,
        help="versioned hash salt; changing it intentionally reshuffles pairs",
    )
    parser.add_argument(
        "--expand-ambiguous",
        action="store_true",
        help="expand multi-ID mappings into every possible ORPHA pair",
    )
    parser.add_argument(
        "--include-pairs-without-features",
        action="store_true",
        help="do not reject ORPHA IDs absent from generated diseases.parquet",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    test_ratio = 1.0 - args.train_ratio - args.validation_ratio
    if test_ratio < -1e-12:
        print(
            "error: --train-ratio and --validation-ratio must sum to at most 1",
            file=sys.stderr,
        )
        return 2
    if not args.salt:
        print("error: --salt must not be empty", file=sys.stderr)
        return 2

    input_path = args.input.expanduser().resolve()
    feature_dir = args.feature_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    try:
        if not input_path.is_file():
            raise FileNotFoundError(f"Literature evidence CSV not found: {input_path}")
        csv_bytes = input_path.read_bytes()
        input_hash = hashlib.sha256(csv_bytes).hexdigest()
        feature_ids = (
            None
            if args.include_pairs_without_features
            else load_feature_orpha_ids(feature_dir)
        )
        splits, rejected, fieldnames, counts = split_rows(
            csv_bytes,
            feature_orpha_ids=feature_ids,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            salt=args.salt,
            expand_ambiguous=args.expand_ambiguous,
        )
        manifest = _build_manifest(
            input_path=input_path,
            input_hash=input_hash,
            feature_dir=feature_dir,
            feature_filter_enabled=feature_ids is not None,
            output_dir=output_dir,
            splits=splits,
            rejected=rejected,
            counts=counts,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            salt=args.salt,
            expand_ambiguous=args.expand_ambiguous,
        )
        if manifest["integrity"]["pair_overlap_count"]:
            raise RuntimeError("Internal error: an ORPHA pair spans multiple splits.")
        write_outputs(
            output_dir,
            splits=splits,
            rejected=rejected,
            source_fieldnames=fieldnames,
            manifest=manifest,
        )
    except (FileNotFoundError, RuntimeError, ValueError, OSError, UnicodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"Created stable ORPHA-pair splits in {output_dir}")
    for split_name in SPLIT_NAMES:
        statistics = manifest["splits"][split_name]
        print(
            f"  {split_name}: {statistics['rows']:,} rows, "
            f"{statistics['unique_pairs']:,} pairs, "
            f"{statistics['unique_papers']:,} papers"
        )
    print(f"  rejected: {len(rejected):,} observations")
    print(
        "  pair overlap: "
        f"{manifest['integrity']['pair_overlap_count']}; "
        "papers spanning splits: "
        f"{manifest['integrity']['papers_spanning_multiple_splits']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
