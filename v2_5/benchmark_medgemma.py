"""Run the leakage-safe paper-input benchmark with v2.5 MedGemma extraction.

Each query/level extraction is written atomically and cached independently, so
an interrupted run resumes without repeating completed model calls.

Examples:

    python -m v2_5.benchmark_medgemma --levels first_2000_words
    python -m v2_5.benchmark_medgemma --levels title_abstract,first_500_words,first_2000_words
    python -m v2_5.benchmark_medgemma --score-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "v2"
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

from benchmark_paper_input import (
    DATABASE,
    LEVELS,
    SELECTED_CONFIG,
    WORD,
    json_safe,
    paper_text,
    relation_neighbors,
    score_records,
    select_query_papers,
    summarize,
    write_csv,
)
from data_sources import load_bundle
from modalities import DiseaseEncoder
from scoring import PatternFusion, SimilarityEngine, build_training_set, fit_logistic

from .client import ClientConfig, MedGemmaClient
from .extraction import ExtractionResult, HybridPaperExtractor, PROMPT_VERSION, save_extraction
from .transformers_client import TransformersConfig, TransformersMedGemmaClient


OUTPUT_DIR = ROOT / ".data" / "v2_5" / "paper_input_benchmark"
LEVEL_NAMES = tuple(name for name, _ in LEVELS)


def parse_levels(value: str) -> list[str]:
    levels = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(levels) - set(LEVEL_NAMES))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown levels: {unknown}; choose from {LEVEL_NAMES}")
    return levels


def extraction_path(output_dir: Path, target_id: str, level: str) -> Path:
    safe_target = target_id.replace(":", "_")
    return output_dir / "extractions" / level / f"{safe_target}.json"


def load_extraction(path: Path) -> ExtractionResult:
    document = json.loads(path.read_text(encoding="utf-8"))
    document.pop("provenance", None)
    document.pop("source_level_word_count", None)
    return ExtractionResult.from_dict(document)


def record_counts(record: dict[str, Any]) -> dict[str, int]:
    return {
        "genes": len(record.get("genes", {})),
        "phenotypes": len(record.get("phenotypes", {})),
        "drugs": len(record.get("drugs", {})),
        "inheritance": len(record.get("inheritance", [])),
        "onset": len(record.get("onset", [])),
        "prevalence": int(record.get("prevalence") is not None),
    }


def write_progress(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, indent=2), encoding="utf-8")
    temporary.replace(path)


def build_cohort(bundle: Any, database: Path, max_queries: int) -> tuple[list[Any], dict[str, set[str]], dict[str, Any]]:
    curated_neighbors, relations = relation_neighbors(bundle)
    candidates = {
        disease
        for disease in bundle.records
        if bundle.meta["split"][disease] == "test" and curated_neighbors.get(disease)
    }
    papers = select_query_papers(database, candidates, max_queries)
    changed = True
    while changed:
        query_set = {paper.target_id for paper in papers}
        kept = [
            paper
            for paper in papers
            if curated_neighbors.get(paper.target_id, set()) - query_set
        ]
        changed = len(kept) != len(papers)
        papers = kept
    return papers, curated_neighbors, relations


def fit_heldout_model(
    bundle: Any,
    query_ids: list[str],
    relations: dict[str, Any],
    selected_config: Path,
) -> tuple[DiseaseEncoder, SimilarityEngine, list[str], PatternFusion]:
    query_set = set(query_ids)
    train_ids = {
        disease for disease in bundle.records if bundle.meta["split"][disease] == "train"
    }
    gallery_ids = sorted(set(bundle.records) - query_set)
    if query_set & train_ids or query_set & set(gallery_ids):
        raise AssertionError("query-disease leakage into training or gallery")
    encoder = DiseaseEncoder(bundle.knowledge)
    encoder.fit(
        [bundle.records[disease] for disease in sorted(train_ids)],
        [bundle.records[disease] for disease in gallery_ids],
    )
    matrices = encoder.transform([bundle.records[disease] for disease in gallery_ids])
    engine = SimilarityEngine(matrices, gallery_ids)
    config = json.loads(selected_config.read_text(encoding="utf-8"))
    training = build_training_set(
        engine,
        list(relations.values()),
        train_ids,
        negatives_per_positive=int(config["negatives_per_positive"]),
        seed=int(config["seed"]),
    )
    params = config["logistic_params"]
    base = fit_logistic(
        training,
        engine.modalities,
        C=float(params["C"]),
        use_availability=bool(params["use_availability"]),
        nonnegative=bool(params.get("nonnegative", True)),
        name=str(config["logistic"]),
    )
    return encoder, engine, gallery_ids, PatternFusion(training, base)


def comparison_report(
    output_dir: Path,
    metadata: dict[str, Any],
    summaries: list[dict[str, Any]],
    deterministic_summary_path: Path,
) -> None:
    medgemma = {(row["variant"], row["level"]): row for row in summaries}
    deterministic: dict[tuple[str, str], dict[str, Any]] = {}
    if deterministic_summary_path.is_file():
        source = json.loads(deterministic_summary_path.read_text(encoding="utf-8"))
        deterministic = {
            (row["variant"], row["level"]): row
            for row in source.get("conditions", [])
        }
    lines = [
        "# V2.5 MedGemma paper-input benchmark",
        "",
        f"Generated {metadata['generated_at']}.",
        "",
        f"- Query diseases: **{metadata['query_count']}**",
        f"- Unique papers: **{metadata['unique_paper_count']}**",
        f"- Model: `{metadata['model']}`",
        f"- Prompt version: `{metadata['prompt_version']}`",
        f"- Completed extraction calls: **{metadata['completed_extractions']}**",
        f"- Cache hits: **{metadata['cache_hits']}**",
        f"- Deterministic fallbacks: **{metadata['fallbacks']}**",
        "",
        "## Curated-neighbour results",
        "",
        "| Available paper information | Extractor | MAP | MRR | Hits@10 | Recall@50 | nDCG@10 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for level in metadata["levels"]:
        baseline = deterministic.get(("masked_extracted", level))
        if baseline:
            lines.append(
                f"| {level} | deterministic v2 | {baseline['curated_map']:.4f} | "
                f"{baseline['curated_mrr']:.4f} | {100 * baseline['curated_hits_at_10']:.1f}% | "
                f"{100 * baseline['curated_recall_at_50']:.1f}% | {baseline['curated_ndcg_at_10']:.4f} |"
            )
        hybrid = medgemma.get(("medgemma_hybrid", level))
        if hybrid:
            lines.append(
                f"| {level} | v2.5 MedGemma hybrid | {hybrid['curated_map']:.4f} | "
                f"{hybrid['curated_mrr']:.4f} | {100 * hybrid['curated_hits_at_10']:.1f}% | "
                f"{100 * hybrid['curated_recall_at_50']:.1f}% | {hybrid['curated_ndcg_at_10']:.4f} |"
            )
    lines.extend(
        [
            "",
            "All query diseases were jointly excluded from model fitting, encoder catalogues, and retrieval.",
            "MedGemma features are automated, evidence-grounded, and unverified.",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database", type=Path, default=DATABASE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--selected-config", type=Path, default=SELECTED_CONFIG)
    parser.add_argument("--levels", type=parse_levels, default=list(LEVEL_NAMES))
    parser.add_argument("--max-queries", type=int, default=200)
    parser.add_argument("--max-prompt-words", type=int, default=2_000)
    parser.add_argument(
        "--backend",
        choices=["endpoint", "transformers"],
        default=os.environ.get("MEDGEMMA_BACKEND", "endpoint"),
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--quantization", choices=["4bit", "8bit", "bf16", "fp16"], default=None)
    parser.add_argument("--compute-dtype", choices=["auto", "bfloat16", "float16"], default=None)
    parser.add_argument("--max-input-tokens", type=int, default=8_192)
    parser.add_argument("--model-cache-dir", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-output-tokens", type=int, default=4_096)
    parser.add_argument("--response-mode", choices=["json_schema", "json_object", "prompt_only"], default=None)
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--refresh", action="store_true", help="Ignore completed extraction files and endpoint cache.")
    parser.add_argument("--allow-fallback", action="store_true", help="Score deterministic fallback records if LLM calls fail.")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument(
        "--deterministic-summary",
        type=Path,
        default=V2 / ".data" / "paper_input_benchmark" / "summary.json",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.extract_only and args.score_only:
        raise ValueError("--extract-only and --score-only cannot be combined")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle()
    papers, curated_neighbors, relations = build_cohort(bundle, args.database, args.max_queries)
    query_ids = [paper.target_id for paper in papers]
    query_set = set(query_ids)
    progress_path = output_dir / "progress.json"

    if args.backend == "transformers":
        config: Any = TransformersConfig.from_environment(
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
        client: Any = TransformersMedGemmaClient(config)
    else:
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
        client = MedGemmaClient(config)
    extractor = HybridPaperExtractor(
        bundle.knowledge,
        client,
        max_input_words=args.max_prompt_words,
        strict_llm=not args.allow_fallback,
    )

    results: dict[tuple[str, str], ExtractionResult] = {}
    completed = 0
    total = len(papers) * len(args.levels)
    if not args.score_only:
        for level in args.levels:
            for paper in papers:
                path = extraction_path(output_dir, paper.target_id, level)
                if path.is_file() and not args.refresh:
                    result = load_extraction(path)
                else:
                    text = paper_text(paper, level, masked=True)
                    result = extractor.extract(
                        text,
                        focal_hint=(
                            "The focal disease is the primary disorder studied by this paper. "
                            "All recognized disease names are masked as DISEASE."
                        ),
                        source_id=f"paper:{paper.paper_id}:{paper.target_id}:{level}",
                        use_cache=not args.refresh,
                    )
                    save_extraction(
                        path,
                        result,
                        provenance={
                            "target_id": paper.target_id,
                            "paper_id": paper.paper_id,
                            "pmid": paper.pmid,
                            "pmcid": paper.pmcid,
                            "doi": paper.doi,
                            "level": level,
                            "source_level_word_count": len(WORD.findall(text)),
                        },
                    )
                if result.status == "deterministic_fallback" and not args.allow_fallback:
                    raise RuntimeError(f"Unexpected LLM fallback for {paper.target_id}/{level}")
                results[(paper.target_id, level)] = result
                completed += 1
                write_progress(
                    progress_path,
                    {
                        "completed": completed,
                        "total": total,
                        "last_target": paper.target_id,
                        "last_level": level,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                if completed % 10 == 0 or completed == total:
                    print(f"Completed MedGemma extractions: {completed}/{total}", flush=True)
    else:
        for level in args.levels:
            for paper in papers:
                path = extraction_path(output_dir, paper.target_id, level)
                if not path.is_file():
                    raise FileNotFoundError(f"Missing extraction for --score-only: {path}")
                results[(paper.target_id, level)] = load_extraction(path)
        completed = total

    statuses = [result.status for result in results.values()]
    metadata: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "query_count": len(query_ids),
        "unique_paper_count": len({paper.paper_id for paper in papers}),
        "levels": args.levels,
        "max_prompt_words": args.max_prompt_words,
        "model": config.model,
        "model_revision": config.model_revision,
        "backend": args.backend,
        "base_url": getattr(config, "base_url", "in-process"),
        "response_mode": getattr(config, "response_mode", "prompt_json_validation"),
        "quantization": getattr(config, "quantization", ""),
        "prompt_version": PROMPT_VERSION,
        "completed_extractions": completed,
        "cache_hits": sum(result.cached for result in results.values()),
        "fallbacks": sum(status == "deterministic_fallback" for status in statuses),
        "accepted_features": sum(len(result.accepted_evidence) for result in results.values()),
        "rejected_features": sum(len(result.rejected_features) for result in results.values()),
        "query_ids_sha256": hashlib.sha256("\n".join(query_ids).encode("utf-8")).hexdigest(),
    }
    if args.extract_only:
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata

    encoder, engine, gallery_ids, fusion = fit_heldout_model(
        bundle, query_ids, relations, args.selected_config
    )
    gallery_set = set(gallery_ids)
    curated_gold = {
        target: curated_neighbors.get(target, set()) & gallery_set for target in query_ids
    }
    paper_gold = {
        paper.target_id: paper.paper_comparators & gallery_set for paper in papers
    }
    rows: list[dict[str, Any]] = []
    for level in args.levels:
        level_results = [results[(target, level)] for target in query_ids]
        records = [result.record for result in level_results]
        words = [sum(int(passage["word_count"]) for passage in result.passages) for result in level_results]
        counts = [record_counts(record) for record in records]
        level_rows = score_records(
            records,
            query_ids,
            encoder,
            engine,
            gallery_ids,
            fusion,
            curated_gold,
            paper_gold,
            condition="v2_5",
            level=level,
            variant="medgemma_hybrid",
            words=words,
            extraction_counts=counts,
        )
        for row, result in zip(level_rows, level_results):
            row.update(
                {
                    "llm_status": result.status,
                    "llm_cached": int(result.cached),
                    "llm_accepted": len(result.accepted_evidence),
                    "llm_rejected": len(result.rejected_features),
                    "llm_model": result.model,
                }
            )
        rows.extend(level_rows)
    summaries = summarize(rows)
    metadata.update(
        {
            "gallery_count": len(gallery_ids),
            "holdout_verified": not query_set & set(gallery_ids),
            "paper_gold_queries": sum(bool(value) for value in paper_gold.values()),
        }
    )
    write_csv(output_dir / "per_query.csv", rows)
    (output_dir / "summary.json").write_text(
        json.dumps(json_safe({"metadata": metadata, "conditions": summaries}), indent=2),
        encoding="utf-8",
    )
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    comparison_report(output_dir, metadata, summaries, args.deterministic_summary)
    print(f"Wrote v2.5 benchmark to {output_dir}", flush=True)
    return metadata


if __name__ == "__main__":
    run(build_parser().parse_args())

