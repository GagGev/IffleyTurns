"""Paths and deterministic split utilities for the v4 embedding model."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V4_ROOT = PROJECT_ROOT / "v4"
FEATURE_DIR = PROJECT_ROOT / ".data" / "features"
DATABASE_DIR = PROJECT_ROOT / ".data" / "databases"
DATA_DIR = V4_ROOT / ".data"
CACHE_DIR = DATA_DIR / "cache"
MODEL_DIR = DATA_DIR / "model"
EVALUATION_DIR = DATA_DIR / "evaluation"
MODEL_PATH = MODEL_DIR / "embedding_model.joblib"

SPLIT_SALT = "rare-disease-v2-disease-split"
SPLIT_RATIOS = {"train": 0.6, "validation": 0.2, "test": 0.2}


def assign_split(identifier: str, salt: str = SPLIT_SALT) -> str:
    """Use the same stable disease-level split as v2."""

    digest = hashlib.blake2b(f"{salt}\0{identifier}".encode(), digest_size=8).digest()
    fraction = int.from_bytes(digest, "big") / 2**64
    cumulative = 0.0
    for name, ratio in SPLIT_RATIOS.items():
        cumulative += ratio
        if fraction < cumulative:
            return name
    return "test"


def normalize_curie(value: str) -> str:
    value = str(value).strip()
    if ":" not in value and "_" in value:
        prefix, _, local = value.partition("_")
        return f"{prefix}:{local}"
    return value


def write_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(document, indent=2, default=_json_default), encoding="utf-8")
    tmp.replace(path)


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


@contextmanager
def timed(message: str) -> Iterator[None]:
    start = time.perf_counter()
    print(f"[v4] {message}...", flush=True)
    yield
    print(f"[v4] {message} done in {time.perf_counter() - start:.1f}s", flush=True)


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
