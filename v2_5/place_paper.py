"""Extract a disease profile from a paper with MedGemma and rank it with v2.

Examples:

    python -m v2_5.place_paper --input paper.txt --disease-name "New disorder"
    python -m v2_5.place_paper --paper-id 123 --top 20 --output result.json
    python -m v2_5.place_paper --input paper.pdf --deterministic-only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "v2"
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

from place_disease import place, print_results, slug
from production import load_model

from .client import ClientConfig, JsonCompletionClient, MedGemmaClient
from .extraction import HybridPaperExtractor
from .passages import SourceUnit, units_from_text
from .transformers_client import TransformersConfig, TransformersMedGemmaClient


DATABASE = ROOT / ".data" / "literature_acquisition" / "papers.sqlite"


def _local_tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1].lower()


def _element_text(element: ET.Element) -> str:
    return re.sub(r"\s+", " ", " ".join(element.itertext())).strip()


def units_from_xml(path: Path) -> list[SourceUnit]:
    root = ET.parse(path).getroot()
    units: list[SourceUnit] = []
    article_titles = [element for element in root.iter() if _local_tag(element) == "article-title"]
    if article_titles:
        units.append(SourceUnit("Title", "xml:article-title", _element_text(article_titles[0])))
    abstracts = [element for element in root.iter() if _local_tag(element) == "abstract"]
    if abstracts:
        units.append(SourceUnit("Abstract", "xml:abstract", _element_text(abstracts[0])))
    paragraph_number = 0
    for element in root.iter():
        if _local_tag(element) != "p":
            continue
        text = _element_text(element)
        if not text:
            continue
        paragraph_number += 1
        units.append(SourceUnit("Body", f"xml:p:{paragraph_number}", text))
    return units


def units_from_pdf(path: Path) -> list[SourceUnit]:
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError("PDF input requires `pip install pypdf`; text, XML, and database inputs need no extra package") from error
    units: list[SourceUnit] = []
    for page_number, page in enumerate(PdfReader(str(path)).pages, 1):
        text = page.extract_text() or ""
        for paragraph_number, paragraph in enumerate(re.split(r"\n\s*\n+", text), 1):
            paragraph = re.sub(r"\s+", " ", paragraph).strip()
            if paragraph:
                units.append(
                    SourceUnit(
                        "Paper",
                        f"pdf:page:{page_number}:paragraph:{paragraph_number}",
                        paragraph,
                    )
                )
    return units


def units_from_json(path: Path) -> tuple[list[SourceUnit], dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Paper JSON must be an object")
    units: list[SourceUnit] = []
    for field, section in (("title", "Title"), ("abstract", "Abstract")):
        if document.get(field):
            units.append(SourceUnit(section, f"json:{field}", str(document[field])))
    body = document.get("body", document.get("text", ""))
    if isinstance(body, list):
        for index, value in enumerate(body, 1):
            if isinstance(value, dict):
                units.append(
                    SourceUnit(
                        str(value.get("section") or "Body"),
                        str(value.get("locator") or f"json:body:{index}"),
                        str(value.get("text") or ""),
                    )
                )
            elif str(value).strip():
                units.append(SourceUnit("Body", f"json:body:{index}", str(value)))
    elif str(body).strip():
        units.extend(units_from_text(str(body)))
    return units, document


def units_from_database(paper_id: int, database: Path = DATABASE) -> tuple[list[SourceUnit], dict[str, Any]]:
    with sqlite3.connect(database) as db:
        db.row_factory = sqlite3.Row
        paper = db.execute("SELECT * FROM papers WHERE paper_id=?", (paper_id,)).fetchone()
        if paper is None:
            raise ValueError(f"Paper {paper_id} is not in {database}")
        units = [SourceUnit("Title", "title", str(paper["title"] or ""))]
        if paper["abstract"]:
            units.append(SourceUnit("Abstract", "abstract", str(paper["abstract"])))
        units.extend(
            SourceUnit(str(row["section_name"]), str(row["source_locator"]), str(row["text"]))
            for row in db.execute(
                "SELECT section_name,source_locator,text FROM fulltext_sections WHERE paper_id=? ORDER BY section_id",
                (paper_id,),
            )
        )
        provenance = {
            "paper_id": paper_id,
            "pmid": paper["pmid"],
            "pmcid": paper["pmcid"],
            "doi": paper["doi"],
            "title": paper["title"],
            "access_status": paper["access_status"],
            "source_urls": json.loads(paper["source_urls_json"] or "[]"),
            "full_text_raw_path": paper["full_text_raw_path"],
        }
    return units, provenance


def load_source(path: Path) -> tuple[list[SourceUnit], dict[str, Any]]:
    suffix = path.suffix.casefold()
    if suffix in {".xml", ".nxml"}:
        units = units_from_xml(path)
        metadata: dict[str, Any] = {}
    elif suffix == ".pdf":
        units = units_from_pdf(path)
        metadata = {}
    elif suffix == ".json":
        units, metadata = units_from_json(path)
    else:
        units = units_from_text(path.read_text(encoding="utf-8"))
        metadata = {}
    metadata.update({"input_path": str(path.resolve()), "input_format": suffix or "text"})
    return units, metadata


def build_client(args: argparse.Namespace) -> JsonCompletionClient | None:
    if args.deterministic_only:
        return None
    if args.backend == "transformers":
        config = TransformersConfig.from_environment(
            model=args.model,
            model_revision=args.model_revision,
            hf_token=args.hf_token,
            quantization=args.quantization,
            compute_dtype=args.compute_dtype,
            max_input_tokens=args.max_input_tokens,
            max_output_tokens=args.max_output_tokens,
            model_cache_dir=args.model_cache_dir,
            response_cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
        )
        return TransformersMedGemmaClient(config)
    config = ClientConfig.from_environment(
        base_url=args.base_url,
        model=args.model,
        model_revision=args.model_revision,
        api_key=args.api_key,
        timeout_seconds=args.timeout,
        max_output_tokens=args.max_output_tokens,
        response_mode=args.response_mode,
        allow_remote=args.allow_remote,
        cache_dir=args.cache_dir,
    )
    return MedGemmaClient(config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Paper as text, JSON, XML/JATS, or PDF.")
    source.add_argument("--paper-id", type=int, help="Paper ID in the acquired SQLite corpus.")
    parser.add_argument("--database", type=Path, default=DATABASE)
    parser.add_argument("--disease-name", default="", help="Display name or focal-disease hint.")
    parser.add_argument("--query-id", default="", help="Identifier used to exclude an existing gallery node.")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--max-input-words", type=int, default=2_000)
    parser.add_argument("--use-name-modality", action="store_true", help="Allow the supplied disease name to affect ranking.")
    parser.add_argument("--deterministic-only", action="store_true", help="Skip MedGemma and use the v2 exact extractor.")
    parser.add_argument("--strict-llm", action="store_true", help="Fail instead of falling back when MedGemma is unavailable.")
    parser.add_argument(
        "--backend",
        choices=["endpoint", "transformers"],
        default=os.environ.get("MEDGEMMA_BACKEND", "endpoint"),
        help="Use a separate OpenAI-compatible endpoint or load MedGemma directly in this process.",
    )
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint; default MEDGEMMA_BASE_URL or localhost.")
    parser.add_argument("--model", default=None, help="Endpoint model ID; default MEDGEMMA_MODEL.")
    parser.add_argument("--model-revision", default=None, help="Immutable weight revision for provenance.")
    parser.add_argument("--api-key", default=None, help="Endpoint key; default MEDGEMMA_API_KEY.")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token; prefer the HF_TOKEN environment variable.")
    parser.add_argument("--quantization", choices=["4bit", "8bit", "bf16", "fp16"], default=None)
    parser.add_argument("--compute-dtype", choices=["auto", "bfloat16", "float16"], default=None)
    parser.add_argument("--max-input-tokens", type=int, default=8_192)
    parser.add_argument("--model-cache-dir", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-output-tokens", type=int, default=2_048)
    parser.add_argument("--response-mode", choices=["json_schema", "json_object", "prompt_only"], default=None)
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--output", type=Path, help="Write extraction, provenance, and neighbours as JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.input:
        units, provenance = load_source(args.input)
        source_id = str(args.input.resolve())
        default_name = args.input.stem
    else:
        units, provenance = units_from_database(args.paper_id, args.database)
        source_id = f"paper:{args.paper_id}"
        default_name = str(provenance.get("title") or f"paper-{args.paper_id}")
    if not units:
        raise ValueError("No text could be extracted from the paper")

    model = load_model()
    extractor = HybridPaperExtractor(
        model.knowledge,
        build_client(args),
        max_input_words=args.max_input_words,
        strict_llm=args.strict_llm,
    )
    focal_hint = (
        f"The focal disease is {args.disease_name}. Extract only facts asserted for it."
        if args.disease_name
        else ""
    )
    extraction = extractor.extract(
        units=units,
        focal_hint=focal_hint,
        source_id=source_id,
        use_cache=not args.no_cache,
    )
    display_name = args.disease_name or extraction.focal_disease_label or default_name
    extraction.record["name"] = display_name if args.use_name_modality else ""
    query_id = args.query_id or f"USER:{slug(display_name)}"
    neighbors, scored = place(model, extraction.record, query_id, top=args.top)
    print(f"V2.5 extraction status: {extraction.status}")
    print(
        f"Accepted {len(extraction.accepted_evidence)} MedGemma features; "
        f"rejected {len(extraction.rejected_features)}; "
        f"selected {sum(item['word_count'] for item in extraction.passages)} paper words."
    )
    for warning in extraction.warnings:
        print(f"warning: {warning}")
    print_results(display_name, neighbors, scored, hide=())

    if args.output:
        document = extraction.to_dict()
        document.update(
            {
                "query_id": query_id,
                "display_name": display_name,
                "provenance": provenance,
                "neighbors": neighbors,
            }
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(args.output)
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

