"""Place an uploaded paper with v2_5: MedGemma extracts the disease profile, v2 places it.

This is ``python -m v2_5.place_paper --input <file>`` run inside the API
server, so the v2 model that is already loaded is reused and the result is
returned to the browser in the same shape ``--output`` writes.  MedGemma is
reached through the local OpenAI-compatible endpoint (medgemma_server.py).
If it is unavailable, v2_5 falls back to its deterministic extractor and says
so in ``status`` and ``warnings``.
"""

from __future__ import annotations

import base64
import binascii
import dataclasses
import hashlib
import json
import os
import re
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / ".data" / "v2_5" / "medgemma_cache"
ALLOWED_SUFFIXES = {".pdf", ".txt", ".md", ".xml", ".nxml", ".json"}
MAX_PAPER_BYTES = 20_000_000
DEFAULT_BASE_URL = "http://127.0.0.1:8766/v1"
# Words of the paper sent to MedGemma; v2_5 picks the most informative passages within this budget.
MAX_INPUT_WORDS = 1_000


def served_model(base_url: str) -> str:
    """The model ID the endpoint reports, for provenance."""

    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=3) as response:
            data = json.load(response).get("data") or []
            return str(data[0]["id"]) if data else "medgemma"
    except Exception:
        return "medgemma"


def title_of(units: list[Any], fallback: str) -> str:
    for unit in units:
        if str(getattr(unit, "section", "")).lower() == "title" and getattr(unit, "text", "").strip():
            return unit.text.strip()
    for unit in units[:3]:
        line = getattr(unit, "text", "").strip().split("\n")[0]
        if 12 <= len(line) <= 200:
            return line
    return fallback


# Words too general to show that a quote is about a particular feature.
GENERAL_WORDS = {"abnormal", "abnormality", "disease", "disorder", "syndrome", "increased", "decreased", "onset"}


def names_feature(item: dict[str, Any]) -> bool:
    """Whether an evidence quote contains the feature it is offered for: a gene's symbol, or every distinctive word
    of any other feature's label.  A small model sometimes quotes a nearby sentence that does not mention it."""

    quote = str(item.get("quote", "")).lower()
    if item.get("feature_type") == "gene":
        return bool(re.search(rf"\b{re.escape(str(item.get('identifier', '')).lower())}\b", quote))
    words = [w for w in re.findall(r"[a-z]{4,}", str(item.get("label", "")).lower()) if w not in GENERAL_WORDS]
    return bool(words) and all(re.search(rf"\b{re.escape(w[:6])}", quote) for w in words)


# A 4B model left to itself copies whole paragraphs as "quotes" and runs out of output tokens mid-way.
COMPACT_RULES = (
    "\n\nKeep the output short. Every quote must be at most 15 consecutive words copied exactly from its passage, "
    "and must contain the words that state that feature (for example the name of the sign or gene), not just nearby text. "
    "List at most 15 phenotypes and at most 5 items in every other list. Prefer empty lists to long output."
)


class CompactClient:
    """Wraps v2_5's endpoint client for a small local MedGemma.

    It asks for short quotes and few items, and rejects a reply that is not the extraction object (for example one
    cut off at the token limit, from which only an inner item parses), so v2_5 falls back to its deterministic
    extractor instead of reporting success with nothing extracted.
    """

    def __init__(self, inner: Any):
        self.inner = inner
        self.config = inner.config

    def complete_json(self, system_prompt, user_prompt, schema, prompt_version, use_cache=True):
        from v2_5.client import InvalidModelResponse

        completion = self.inner.complete_json(
            system_prompt + COMPACT_RULES, user_prompt, schema, f"{prompt_version}+compact", use_cache=use_cache
        )
        properties = (schema or {}).get("properties", {})
        document = completion.document
        if not isinstance(document, dict) or not set(properties) & set(document):
            raise InvalidModelResponse(
                "MedGemma's reply was not a complete extraction (it may have run out of output tokens)."
            )
        # The model writes null for a list it has nothing for; v2_5 expects an empty list.
        fixed = {
            key: ([] if value is None and properties.get(key, {}).get("type") == "array" else value)
            for key, value in document.items()
        }
        return dataclasses.replace(completion, document=fixed)


class PaperPlacer:
    def __init__(
        self,
        knowledge: Callable[[], Any],
        place_record: Callable[[dict[str, Any], str, int], list[dict[str, Any]]],
        base_url: Optional[str] = None,
    ):
        self.knowledge = knowledge
        self.place_record = place_record
        self.base_url = (base_url or os.environ.get("MEDGEMMA_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")

    def place(self, body: dict[str, Any]) -> dict[str, Any]:
        filename = Path(str(body.get("filename") or "paper.txt")).name
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise ValueError(f"Unsupported file type {suffix or '(none)'}; use PDF, text, Markdown, XML or JSON.")
        try:
            content = base64.b64decode(str(body.get("content") or ""), validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError("The file content must be base64-encoded.") from error
        if not content:
            raise ValueError("The file is empty.")
        if len(content) > MAX_PAPER_BYTES:
            raise ValueError("The file is larger than 20 MB.")
        disease_name = str(body.get("disease_name") or "").strip()
        top = max(1, min(int(body.get("top") or 20), 50))

        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from v2_5.client import ClientConfig, MedGemmaClient
        from v2_5.extraction import HybridPaperExtractor
        from v2_5.place_paper import load_source

        from place_disease import slug  # v2's; importing v2_5.place_paper puts v2 on the path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"paper{suffix}"
            path.write_bytes(content)
            units, provenance = load_source(path)
        if not units:
            raise ValueError("No text could be read from the file. A scanned PDF needs OCR first.")
        provenance = {k: v for k, v in provenance.items() if k != "input_path"}
        provenance.update({"input_path": filename, "title": provenance.get("title") or title_of(units, Path(filename).stem)})

        client = CompactClient(
            MedGemmaClient(
                ClientConfig(
                    base_url=self.base_url,
                    model=served_model(self.base_url),
                    timeout_seconds=900,
                    max_output_tokens=2_048,
                    response_mode="json_schema",
                    cache_dir=CACHE_DIR,
                )
            )
        )
        extractor = HybridPaperExtractor(self.knowledge(), client, max_input_words=MAX_INPUT_WORDS, strict_llm=False)
        focal_hint = (
            f"The focal disease is {disease_name}. Extract only facts asserted for it." if disease_name else ""
        )
        digest = hashlib.sha256(content).hexdigest()[:16]
        extraction = extractor.extract(units=units, focal_hint=focal_hint, source_id=f"upload:{digest}")

        display_name = disease_name or extraction.focal_disease_label or provenance["title"]
        # As place_paper does by default: the name is a label only and does not affect similarity.
        extraction.record["name"] = ""
        query_id = f"USER:{slug(display_name)}"
        neighbors = self.place_record(extraction.record, query_id, top)

        document = extraction.to_dict()
        for item in document.get("accepted_evidence", []):
            item["quote_names_feature"] = names_feature(item)
        unnamed = [item["label"] for item in document.get("accepted_evidence", []) if not item["quote_names_feature"]]
        if unnamed:
            document.setdefault("warnings", []).append(
                f"The quotes for {', '.join(unnamed)} do not name the feature; check them against the paper."
            )
        document.update(
            {"query_id": query_id, "display_name": display_name, "provenance": provenance, "neighbors": neighbors}
        )
        return document
