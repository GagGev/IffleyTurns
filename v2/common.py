"""Shared paths and small utilities for the v2 rare-disease similarity models."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V2_ROOT = PROJECT_ROOT / "v2"
FEATURE_DIR = PROJECT_ROOT / ".data" / "features"
DATABASE_DIR = PROJECT_ROOT / ".data" / "databases"
V2_DATA_DIR = V2_ROOT / ".data"
CACHE_DIR = V2_DATA_DIR / "cache"
EVALUATION_DIR = V2_DATA_DIR / "evaluation"
MODEL_DIR = V2_DATA_DIR / "model"
GRAPH_DIR = V2_DATA_DIR / "graph"

SPLIT_SALT = "rare-disease-v2-disease-split"
SPLIT_RATIOS = {"train": 0.6, "validation": 0.2, "test": 0.2}


def normalize_orpha_id(value: str | int) -> str:
    """Normalize ``58``, ``ORPHA_58``, or ``ORPHA:58`` to ``ORPHA:58``."""

    match = re.fullmatch(r"\s*(?:(?:ORPHA|ORPHANET)\s*[:_-]?\s*)?(\d+)\s*", str(value), re.I)
    if not match:
        raise ValueError(f"Invalid ORPHA identifier {value!r}; expected e.g. ORPHA:558.")
    return f"ORPHA:{int(match.group(1))}"


def normalize_curie(value: str) -> str:
    """Convert Open Targets style ``MONDO_0001`` identifiers to ``MONDO:0001``."""

    value = str(value).strip()
    if ":" not in value and "_" in value:
        prefix, _, local = value.partition("_")
        return f"{prefix}:{local}"
    return value


def assign_split(identifier: str, salt: str = SPLIT_SALT) -> str:
    """Deterministically assign a disease to train, validation, or test."""

    digest = hashlib.blake2b(f"{salt}\0{identifier}".encode("utf-8"), digest_size=8).digest()
    fraction = int.from_bytes(digest, "big") / 2**64
    cumulative = 0.0
    for name, ratio in SPLIT_RATIOS.items():
        cumulative += ratio
        if fraction < cumulative:
            return name
    return "test"


def pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def write_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(document, indent=2, sort_keys=False, default=_json_default), encoding="utf-8")
    tmp.replace(path)


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@contextmanager
def timed(message: str) -> Iterator[None]:
    start = time.perf_counter()
    print(f"[v2] {message}...", flush=True)
    yield
    print(f"[v2] {message} done in {time.perf_counter() - start:.1f}s", flush=True)


def configure_stdout() -> None:
    """Avoid UnicodeEncodeError on Windows consoles for disease/drug names."""

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def dump_mapping_sizes(mapping: Mapping[str, Any]) -> dict[str, int]:
    return {key: len(value) for key, value in mapping.items()}
