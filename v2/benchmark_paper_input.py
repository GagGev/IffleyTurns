"""Leakage-safe benchmark for retrieving similar diseases from a paper.

The complete protocol is frozen in ``PAPER_INPUT_BENCHMARK.md``.  This script
selects a deterministic cohort of open-access papers, withholds every query
disease from v2 fitting and retrieval, creates nested paper-text inputs, and
evaluates curated and same-paper neighbours.

Run from the repository root:

    python v2/benchmark_paper_input.py
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import re
import sqlite3
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from benchmarks import build_relations
from data_sources import Bundle, Knowledge, empty_record, load_bundle
from modalities import DiseaseEncoder, INHERITANCE_MAP, MODALITIES, ONSET_BINS
from scoring import PatternFusion, SimilarityEngine, build_training_set, fit_logistic


DATABASE = ROOT / ".data" / "literature_acquisition" / "papers.sqlite"
OUTPUT_DIR = HERE / ".data" / "paper_input_benchmark"
SELECTED_CONFIG = HERE / ".data" / "evaluation" / "selected_config.json"

LEVELS: tuple[tuple[str, int | None], ...] = (
    ("title", None),
    ("title_abstract", None),
    ("first_500_words", 500),
    ("first_2000_words", 2_000),
    ("full_text", 10_000),
)
VARIANTS = ("masked_extracted", "masked_text_only", "natural_extracted")
METRIC_NAMES = ("map", "mrr", "hits_at_10", "recall_at_10", "recall_at_50", "ndcg_at_10")
WORD = re.compile(r"\S+")
TOKEN = re.compile(r"(?<![A-Za-z0-9])([A-Za-z][A-Za-z0-9-]{1,14})(?![A-Za-z0-9])")
ORPHA_LITERAL = re.compile(r"(?i)\bORPHA\s*:?\s*\d+\b")

# Uppercase gene-like strings that are frequent prose rather than reliable
# gene mentions.  Symbols of length 1 are already excluded by TOKEN.
GENE_STOP = {
    "AN", "AND", "AS", "AT", "BE", "BY", "CAN", "DNA", "FOR", "HAS",
    "IN", "IS", "IT", "NO", "NOT", "OF", "ON", "OR", "RNA", "TO", "UP",
}

INHERITANCE_PATTERNS = {
    "autosomal recessive": re.compile(r"\b(?:autosomal[- ]recessive|recessively inherited)\b", re.I),
    "autosomal dominant": re.compile(r"\b(?:autosomal[- ]dominant|dominantly inherited)\b", re.I),
    "x-linked recessive": re.compile(r"\bx[- ]linked recessive\b", re.I),
    "x-linked dominant": re.compile(r"\bx[- ]linked dominant\b", re.I),
    "x-linked": re.compile(r"\bx[- ]linked\b", re.I),
    "mitochondrial inheritance": re.compile(r"\b(?:mitochondrial|maternal) inheritance\b", re.I),
    "oligogenic": re.compile(r"\boligogenic\b", re.I),
    "multigenic/multifactorial": re.compile(r"\b(?:multigenic|multifactorial)\b", re.I),
    "y-linked": re.compile(r"\by[- ]linked\b", re.I),
}
ONSET_PATTERNS = {
    "Antenatal": re.compile(r"\b(?:antenatal|prenatal|fetal|foetal|in utero)\b", re.I),
    "Neonatal": re.compile(r"\b(?:neonatal|newborn|at birth|congenital onset)\b", re.I),
    "Infancy": re.compile(r"\b(?:infancy|infantile|infant-onset|first year of life)\b", re.I),
    "Childhood": re.compile(r"\b(?:childhood|pediatric|paediatric|juvenile)[- ]onset\b", re.I),
    "Adolescent": re.compile(r"\b(?:adolescent|adolescence|teenage)[- ]onset\b", re.I),
    "Adult": re.compile(r"\b(?:adult|adulthood|late)[- ]onset\b", re.I),
    "Elderly": re.compile(r"\b(?:elderly|old age|senile)[- ]onset\b", re.I),
}
PREVALENCE_ONE_IN = re.compile(
    r"\b1\s+(?:in|per)\s+([1-9][\d,]*(?:\.\d+)?)\b", re.I
)
PREVALENCE_PERCENT = re.compile(
    r"\b(?:prevalence|affects?|occurs? in)\D{0,25}(\d+(?:\.\d+)?)\s*%", re.I
)


@dataclass(frozen=True)
class TextUnit:
    section: str
    locator: str
    text: str


@dataclass
class QueryPaper:
    target_id: str
    paper_id: int
    pmid: str
    pmcid: str
    doi: str
    title: str
    abstract: str
    year: int | None
    source_urls: list[str]
    full_text_path: str
    title_mention: bool
    metadata_target_count: int
    paper_comparators: set[str]
    body_chars: int
    units: list[TextUnit]
    mentions: dict[str, list[tuple[int, int]]]


def stable_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_phrase(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", value.casefold()))


def truncate_words(text: str, limit: int) -> str:
    matches = list(WORD.finditer(text))
    if len(matches) <= limit:
        return text.strip()
    return text[: matches[limit - 1].end()].strip()


def mask_spans(text: str, spans: Iterable[tuple[int, int]]) -> str:
    """Mask non-overlapping/overlapping entity spans without changing context."""

    valid = sorted(
        {(max(0, start), min(len(text), end)) for start, end in spans if start < end},
        key=lambda item: (item[0], -item[1]),
    )
    merged: list[list[int]] = []
    for start, end in valid:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    output: list[str] = []
    cursor = 0
    for start, end in merged:
        output.extend((text[cursor:start], " DISEASE "))
        cursor = end
    output.append(text[cursor:])
    return ORPHA_LITERAL.sub(" DISEASE ", "".join(output))


class PhraseMatcher:
    """Small deterministic exact phrase matcher indexed by first token."""

    def __init__(self, entries: Iterable[tuple[str, str]], minimum_chars: int = 3):
        index: dict[str, list[tuple[tuple[str, ...], str]]] = defaultdict(list)
        for phrase, identifier in entries:
            tokens = normalize_phrase(phrase)
            if not tokens or len("".join(tokens)) < minimum_chars:
                continue
            index[tokens[0]].append((tokens, identifier))
        self.index = {
            token: sorted(values, key=lambda value: (-len(value[0]), value[0], value[1]))
            for token, values in index.items()
        }

    def find(self, text: str) -> set[str]:
        tokens = normalize_phrase(text)
        found: set[str] = set()
        for offset, token in enumerate(tokens):
            for phrase, identifier in self.index.get(token, ()):
                if tuple(tokens[offset : offset + len(phrase)]) == phrase:
                    found.add(identifier)
        return found


class PaperExtractor:
    """Convert paper text into the record schema consumed by DiseaseEncoder."""

    def __init__(self, knowledge: Knowledge):
        self.knowledge = knowledge
        phenotype_entries = (
            (label, hpo_id)
            for hpo_id, label in knowledge.hpo_labels.items()
            if label and knowledge.is_phenotypic_abnormality(hpo_id)
        )
        drug_entries = (
            (label, drug_id)
            for drug_id, label in knowledge.drug_labels.items()
            if label and not label.upper().startswith("CHEMBL")
        )
        self.phenotypes = PhraseMatcher(phenotype_entries, minimum_chars=4)
        self.drugs = PhraseMatcher(drug_entries, minimum_chars=4)

    def extract(self, text: str, structured: bool = True) -> tuple[dict[str, Any], dict[str, int]]:
        record = empty_record("")
        record["description"] = text
        counts = {"genes": 0, "phenotypes": 0, "drugs": 0, "inheritance": 0, "onset": 0, "prevalence": 0}
        if not structured:
            return record, counts

        genes = {
            match.group(1)
            for match in TOKEN.finditer(text)
            if match.group(1) in self.knowledge.gene_symbols
            and match.group(1).upper() not in GENE_STOP
        }
        record["genes"] = {gene: 1.0 for gene in sorted(genes)}

        phenotypes = self.phenotypes.find(text)
        record["phenotypes"] = {hpo_id: 0.5 for hpo_id in sorted(phenotypes)}

        drugs = self.drugs.find(text)
        record["drugs"] = {drug_id: 1.0 for drug_id in sorted(drugs)}

        record["inheritance"] = [
            value
            for value, pattern in INHERITANCE_PATTERNS.items()
            if value in INHERITANCE_MAP and pattern.search(text)
        ]
        record["onset"] = [
            value for value, pattern in ONSET_PATTERNS.items() if value in ONSET_BINS and pattern.search(text)
        ]
        record["prevalence"] = extract_prevalence(text)
        for field in ("genes", "phenotypes", "drugs", "inheritance", "onset"):
            counts[field] = len(record[field])
        counts["prevalence"] = int(record["prevalence"] is not None)
        return record, counts


def extract_prevalence(text: str) -> float | None:
    values: list[float] = []
    for match in PREVALENCE_ONE_IN.finditer(text):
        denominator = float(match.group(1).replace(",", ""))
        if denominator >= 10:
            values.append(1.0 / denominator)
    for match in PREVALENCE_PERCENT.finditer(text):
        percentage = float(match.group(1))
        if 0 < percentage <= 100:
            values.append(percentage / 100.0)
    # The rarest explicit estimate is conservative when several population
    # estimates occur in one article.
    return min(values) if values else None


def relation_neighbors(bundle: Bundle) -> tuple[dict[str, set[str]], dict[str, Any]]:
    relations = build_relations(bundle)
    neighbors: dict[str, set[str]] = defaultdict(set)
    for relation in relations.values():
        for disease, related in relation.neighbors.items():
            neighbors[disease].update(related)
    return dict(neighbors), relations


def _json_list(value: str | None) -> list[str]:
    try:
        parsed = json.loads(value or "[]")
        return [str(item) for item in parsed] if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def json_safe(value: Any) -> Any:
    """Replace non-finite floats so result files are strict JSON."""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def select_query_papers(
    db_path: Path,
    candidate_ids: set[str],
    max_queries: int,
) -> list[QueryPaper]:
    """Select one deterministic disease-focused OA paper per candidate disease."""

    with sqlite3.connect(db_path) as db:
        db.row_factory = sqlite3.Row
        papers = {
            int(row["paper_id"]): row
            for row in db.execute(
                """
                SELECT p.*,
                       COALESCE(SUM(LENGTH(s.text)),0) AS body_chars
                FROM papers p
                JOIN fulltext_sections s ON s.paper_id=p.paper_id
                WHERE p.access_status='open_full_text'
                  AND COALESCE(p.title,'')!=''
                GROUP BY p.paper_id
                """
            )
        }
        paper_targets: dict[int, set[str]] = defaultdict(set)
        title_targets: dict[int, set[str]] = defaultdict(set)
        target_mention_counts: dict[tuple[int, str], int] = defaultdict(int)
        for row in db.execute(
            """
            SELECT paper_id,orpha_id,source_locator
            FROM disease_mentions
            WHERE mapping_status='exact' AND orpha_id IS NOT NULL
              AND source_locator IN ('title','abstract')
            """
        ):
            paper_id = int(row["paper_id"])
            target = str(row["orpha_id"])
            paper_targets[paper_id].add(target)
            target_mention_counts[(paper_id, target)] += 1
            if row["source_locator"] == "title":
                title_targets[paper_id].add(target)

        pair_targets: dict[tuple[int, str], set[str]] = defaultdict(set)
        for row in db.execute(
            "SELECT paper_id,orpha_id_a,orpha_id_b FROM paper_pairs"
        ):
            paper_id = int(row["paper_id"])
            left, right = str(row["orpha_id_a"]), str(row["orpha_id_b"])
            pair_targets[(paper_id, left)].add(right)
            pair_targets[(paper_id, right)].add(left)

        choices: dict[str, list[int]] = defaultdict(list)
        for paper_id, targets in paper_targets.items():
            if paper_id not in papers:
                continue
            for target in targets & candidate_ids:
                choices[target].append(paper_id)

        selected: list[tuple[str, int]] = []
        used_papers: set[int] = set()
        target_order = sorted(choices, key=lambda target: (stable_key(target), target))
        for target in target_order:
            ranked_papers = sorted(
                choices[target],
                key=lambda pid: (
                    -int(target in title_targets[pid]),
                    -int(bool(pair_targets.get((pid, target)))),
                    -len(pair_targets.get((pid, target), ())),
                    -target_mention_counts[(pid, target)],
                    -int(papers[pid]["body_chars"] or 0),
                    pid,
                ),
            )
            paper_id = next((pid for pid in ranked_papers if pid not in used_papers), None)
            if paper_id is None:
                continue
            selected.append((target, paper_id))
            used_papers.add(paper_id)
            if len(selected) == max_queries:
                break

        paper_ids = sorted({paper_id for _, paper_id in selected})
        units: dict[int, list[TextUnit]] = defaultdict(list)
        mentions: dict[int, dict[str, list[tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
        if paper_ids:
            placeholders = ",".join("?" for _ in paper_ids)
            for row in db.execute(
                f"""
                SELECT paper_id,section_name,source_locator,text
                FROM fulltext_sections
                WHERE paper_id IN ({placeholders})
                ORDER BY paper_id,section_id
                """,
                paper_ids,
            ):
                units[int(row["paper_id"])].append(
                    TextUnit(str(row["section_name"]), str(row["source_locator"]), str(row["text"]))
                )
            for row in db.execute(
                f"""
                SELECT paper_id,source_locator,char_start,char_end
                FROM disease_mentions
                WHERE paper_id IN ({placeholders})
                ORDER BY paper_id,source_locator,char_start,char_end
                """,
                paper_ids,
            ):
                mentions[int(row["paper_id"])][str(row["source_locator"])].append(
                    (int(row["char_start"]), int(row["char_end"]))
                )

        result: list[QueryPaper] = []
        for target, paper_id in selected:
            row = papers[paper_id]
            target_mentions = title_targets[paper_id]
            result.append(
                QueryPaper(
                    target_id=target,
                    paper_id=paper_id,
                    pmid=str(row["pmid"] or ""),
                    pmcid=str(row["pmcid"] or ""),
                    doi=str(row["doi"] or ""),
                    title=str(row["title"] or ""),
                    abstract=str(row["abstract"] or ""),
                    year=int(row["year"]) if row["year"] is not None else None,
                    source_urls=_json_list(row["source_urls_json"]),
                    full_text_path=str(row["full_text_raw_path"] or ""),
                    title_mention=target in target_mentions,
                    metadata_target_count=len(paper_targets[paper_id]),
                    paper_comparators=set(pair_targets.get((paper_id, target), ())),
                    body_chars=int(row["body_chars"] or 0),
                    units=units[paper_id],
                    mentions=dict(mentions[paper_id]),
                )
            )
    return result


def paper_text(paper: QueryPaper, level: str, masked: bool) -> str:
    def unit_text(locator: str, text: str) -> str:
        return mask_spans(text, paper.mentions.get(locator, ())) if masked else text

    title = unit_text("title", paper.title).strip()
    abstract = unit_text("abstract", paper.abstract).strip()
    body = [unit_text(unit.locator, unit.text).strip() for unit in paper.units]
    if level == "title":
        return title
    if level == "title_abstract":
        return "\n\n".join(value for value in (title, abstract) if value)
    limit = dict(LEVELS)[level]
    stream = "\n\n".join(value for value in (title, abstract, *body) if value)
    return truncate_words(stream, int(limit))


def copy_without_identity(record: dict[str, Any], hide_ontology: bool = True) -> dict[str, Any]:
    result = copy.deepcopy(record)
    result["name"] = ""
    result["synonyms"] = []
    if hide_ontology:
        result["ontology_parents"] = []
    return result


def ranking_metrics(scores: np.ndarray, positives: set[int]) -> dict[str, float]:
    if not positives:
        return {name: math.nan for name in METRIC_NAMES}
    order = np.argsort(-scores, kind="stable")
    relevant = np.fromiter((int(index in positives) for index in order), dtype=np.int8)
    positions = np.flatnonzero(relevant)
    precisions = np.arange(1, len(positions) + 1, dtype=np.float64) / (positions + 1)
    top_10 = relevant[:10]
    top_50 = relevant[:50]
    dcg = float(np.sum(top_10 / np.log2(np.arange(2, len(top_10) + 2))))
    ideal_count = min(len(positives), 10)
    idcg = float(np.sum(1.0 / np.log2(np.arange(2, ideal_count + 2))))
    return {
        "map": float(np.mean(precisions)),
        "mrr": float(1.0 / (positions[0] + 1)),
        "hits_at_10": float(bool(top_10.sum())),
        "recall_at_10": float(top_10.sum() / len(positives)),
        "recall_at_50": float(top_50.sum() / len(positives)),
        "ndcg_at_10": float(dcg / idcg),
    }


def bootstrap_interval(values: Sequence[float], seed: int = 0) -> tuple[float, float]:
    clean = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if len(clean) < 2:
        value = float(clean[0]) if len(clean) else math.nan
        return value, value
    rng = np.random.default_rng(seed)
    samples = rng.choice(clean, size=(2_000, len(clean)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, (0.025, 0.975))
    return float(low), float(high)


def score_records(
    records: list[dict[str, Any]],
    query_ids: list[str],
    encoder: DiseaseEncoder,
    engine: SimilarityEngine,
    gallery_ids: list[str],
    fusion: PatternFusion,
    curated_gold: dict[str, set[str]],
    paper_gold: dict[str, set[str]],
    condition: str,
    level: str,
    variant: str,
    words: list[int],
    extraction_counts: list[dict[str, int]],
) -> list[dict[str, Any]]:
    query_matrices = encoder.transform(records)
    query_available = np.stack(
        [query_matrices[modality].getnnz(axis=1) > 0 for modality in MODALITIES],
        axis=1,
    )
    gallery = engine.gallery(gallery_ids)
    gallery_index = {disease: index for index, disease in enumerate(gallery_ids)}
    rows: list[dict[str, Any]] = []
    batch_size = 16
    for start in range(0, len(records), batch_size):
        stop = min(start + batch_size, len(records))
        matrices = {name: matrix[start:stop] for name, matrix in query_matrices.items()}
        similarities, availability = engine.block(
            matrices, query_available[start:stop], gallery
        )
        scores = fusion.score(similarities, availability)
        for local, target in enumerate(query_ids[start:stop]):
            row_index = start + local
            curated = {
                gallery_index[disease]
                for disease in curated_gold.get(target, ())
                if disease in gallery_index
            }
            same_paper = {
                gallery_index[disease]
                for disease in paper_gold.get(target, ())
                if disease in gallery_index
            }
            row: dict[str, Any] = {
                "target_id": target,
                "condition": condition,
                "variant": variant,
                "level": level,
                "word_count": words[row_index],
                "available_modalities": "|".join(
                    modality
                    for modality, available in zip(MODALITIES, query_available[row_index])
                    if available
                ),
                "curated_gold_count": len(curated),
                "paper_gold_count": len(same_paper),
                **{f"extracted_{key}": value for key, value in extraction_counts[row_index].items()},
            }
            for prefix, positives in (("curated", curated), ("paper", same_paper)):
                metrics = ranking_metrics(scores[local], positives)
                row.update({f"{prefix}_{key}": value for key, value in metrics.items()})
            rows.append(row)
    return rows


def random_rows(
    query_ids: list[str],
    gallery_ids: list[str],
    curated_gold: dict[str, set[str]],
    paper_gold: dict[str, set[str]],
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    gallery_index = {disease: index for index, disease in enumerate(gallery_ids)}
    rows: list[dict[str, Any]] = []
    for target in query_ids:
        scores = rng.random(len(gallery_ids))
        row: dict[str, Any] = {
            "target_id": target,
            "condition": "random",
            "variant": "random",
            "level": "none",
            "word_count": 0,
            "available_modalities": "",
            "curated_gold_count": 0,
            "paper_gold_count": 0,
            **{f"extracted_{key}": 0 for key in ("genes", "phenotypes", "drugs", "inheritance", "onset", "prevalence")},
        }
        for prefix, gold in (("curated", curated_gold), ("paper", paper_gold)):
            positives = {gallery_index[item] for item in gold.get(target, ()) if item in gallery_index}
            row[f"{prefix}_gold_count"] = len(positives)
            row.update(
                {f"{prefix}_{key}": value for key, value in ranking_metrics(scores, positives).items()}
            )
        rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["variant"], row["level"])].append(row)
    result: list[dict[str, Any]] = []
    for (condition, variant, level), group in grouped.items():
        item: dict[str, Any] = {
            "condition": condition,
            "variant": variant,
            "level": level,
            "queries": len(group),
            "median_words": float(np.median([row["word_count"] for row in group])),
        }
        for prefix in ("curated", "paper"):
            valid = [row for row in group if row[f"{prefix}_gold_count"] > 0]
            item[f"{prefix}_queries"] = len(valid)
            item[f"{prefix}_median_gold"] = (
                float(np.median([row[f"{prefix}_gold_count"] for row in valid]))
                if valid
                else math.nan
            )
            for metric in METRIC_NAMES:
                values = [float(row[f"{prefix}_{metric}"]) for row in valid]
                mean = float(np.mean(values)) if values else math.nan
                low, high = bootstrap_interval(
                    values, seed=int(stable_key(f"{condition}:{prefix}:{metric}")[:8], 16)
                )
                item[f"{prefix}_{metric}"] = mean
                item[f"{prefix}_{metric}_ci_low"] = low
                item[f"{prefix}_{metric}_ci_high"] = high
        for field in ("genes", "phenotypes", "drugs", "inheritance", "onset", "prevalence"):
            item[f"queries_with_{field}"] = sum(row[f"extracted_{field}"] > 0 for row in group)
        result.append(item)

    variant_order = {variant: index for index, variant in enumerate(VARIANTS)}
    level_order = {level: index for index, (level, _) in enumerate(LEVELS)}
    result.sort(
        key=lambda row: (
            0 if row["condition"] == "random" else 1 if row["condition"] == "paper" else 2,
            variant_order.get(row["variant"], 99),
            level_order.get(row["level"], 99),
        )
    )
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def percentage(value: float) -> str:
    return "NA" if not math.isfinite(value) else f"{100.0 * value:.1f}%"


def write_report(
    output_dir: Path,
    manifest: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    by_key = {(row["variant"], row["level"]): row for row in summaries}
    lines = [
        "# V2 paper-input benchmark",
        "",
        f"Generated {metadata['generated_at']}. Protocol: `v2/PAPER_INPUT_BENCHMARK.md`.",
        "",
        "## Cohort and leakage control",
        "",
        f"- Query diseases: **{metadata['query_count']}**",
        f"- Unique input papers: **{metadata['unique_paper_count']}**",
        f"- Retrieval gallery diseases: **{metadata['gallery_count']}**",
        f"- Fusion training diseases: **{metadata['train_count']}**",
        f"- Every query disease was jointly excluded from training, encoder catalogue fitting, and the gallery: **{metadata['holdout_verified']}**.",
        f"- Queries with same-paper comparator labels in the gallery: **{metadata['paper_gold_queries']}**",
        "",
        "## Primary results: curated neighbours",
        "",
        "| Input | MAP | MRR | Hits@10 | Recall@10 | Recall@50 | nDCG@10 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    ordered = [("random", "none")]
    ordered.extend(("masked_text_only", level) for level, _ in LEVELS)
    ordered.extend(("masked_extracted", level) for level, _ in LEVELS)
    ordered.extend(("natural_extracted", level) for level, _ in LEVELS)
    ordered.append(("oracle_profile", "database_profile"))
    for variant, level in ordered:
        row = by_key.get((variant, level))
        if not row:
            continue
        label = {
            "random": "Random ranking",
            "masked_text_only": "Masked text only",
            "masked_extracted": "Masked text + extraction",
            "natural_extracted": "Natural text + extraction",
            "oracle_profile": "Oracle structured profile",
        }[variant]
        if level not in {"none", "database_profile"}:
            label += f" — {level.replace('_', ' ')}"
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    f"{row['curated_map']:.4f}",
                    f"{row['curated_mrr']:.4f}",
                    percentage(row["curated_hits_at_10"]),
                    percentage(row["curated_recall_at_10"]),
                    percentage(row["curated_recall_at_50"]),
                    f"{row['curated_ndcg_at_10']:.4f}",
                ]
            )
            + " |"
        )

    primary = by_key.get(("masked_extracted", "full_text"))
    title = by_key.get(("masked_extracted", "title"))
    oracle = by_key.get(("oracle_profile", "database_profile"))
    natural = by_key.get(("natural_extracted", "full_text"))
    lines.extend(["", "## Interpretation", ""])
    if primary and title:
        delta = primary["curated_recall_at_50"] - title["curated_recall_at_50"]
        lines.append(
            f"- In the primary masked/extracted track, curated Recall@50 changed from "
            f"**{percentage(title['curated_recall_at_50'])}** with the title to "
            f"**{percentage(primary['curated_recall_at_50'])}** with full text "
            f"({delta * 100:+.1f} percentage points)."
        )
    if primary and oracle:
        lines.append(
            f"- Full-paper masked extraction reached **{percentage(primary['curated_recall_at_50'])}** "
            f"Recall@50 versus **{percentage(oracle['curated_recall_at_50'])}** for the "
            "database-profile oracle."
        )
    if primary and natural:
        delta = natural["curated_recall_at_50"] - primary["curated_recall_at_50"]
        lines.append(
            f"- Keeping explicit disease names changed full-text Recall@50 by "
            f"**{delta * 100:+.1f} points**; this is a name-assisted sensitivity analysis, "
            "not the strict primary result."
        )
    lines.extend(
        [
            "",
            "## Secondary same-paper comparator results",
            "",
            "These labels come from automated literature pair extraction and are silver, not curated clinical truth.",
            "",
            "| Input | Queries | MAP | Hits@10 | Recall@50 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for variant, level in (
        ("masked_extracted", "title"),
        ("masked_extracted", "title_abstract"),
        ("masked_extracted", "full_text"),
        ("natural_extracted", "full_text"),
        ("oracle_profile", "database_profile"),
    ):
        row = by_key.get((variant, level))
        if not row:
            continue
        lines.append(
            f"| {variant} — {level} | {row['paper_queries']} | {row['paper_map']:.4f} | "
            f"{percentage(row['paper_hits_at_10'])} | {percentage(row['paper_recall_at_50'])} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `query_manifest.csv`: frozen paper cohort and provenance.",
            "- `per_query.csv`: one row per query and condition.",
            "- `summary.json`: aggregate metrics with disease-level bootstrap intervals.",
            "- `metadata.json`: configuration, counts, and leakage checks.",
            "",
            "The benchmark measures retrieval against incomplete relation labels. It is not a "
            "clinical validation and does not imply that unlabelled gallery diseases are irrelevant.",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle()
    curated_neighbors, relations_by_name = relation_neighbors(bundle)
    train_ids = {
        disease for disease in bundle.records if bundle.meta["split"][disease] == "train"
    }
    test_candidates = {
        disease
        for disease in bundle.records
        if bundle.meta["split"][disease] == "test" and curated_neighbors.get(disease)
    }
    papers = select_query_papers(Path(args.database), test_candidates, args.max_queries)

    # Remove queries that would have no curated neighbour once all query diseases
    # are jointly excluded. Repeat to a fixed point because each removal returns a
    # disease to the gallery.
    changed = True
    while changed:
        query_ids_set = {paper.target_id for paper in papers}
        kept = [
            paper
            for paper in papers
            if curated_neighbors.get(paper.target_id, set()) - query_ids_set
        ]
        changed = len(kept) != len(papers)
        papers = kept
    query_ids = [paper.target_id for paper in papers]
    query_set = set(query_ids)
    gallery_ids = sorted(set(bundle.records) - query_set)
    assert not query_set & train_ids
    assert not query_set & set(gallery_ids)

    curated_gold = {
        target: curated_neighbors.get(target, set()) & set(gallery_ids)
        for target in query_ids
    }
    paper_gold = {
        paper.target_id: paper.paper_comparators & set(gallery_ids)
        for paper in papers
    }

    encoder = DiseaseEncoder(bundle.knowledge)
    encoder.fit(
        [bundle.records[disease] for disease in sorted(train_ids)],
        [bundle.records[disease] for disease in gallery_ids],
    )
    gallery_matrices = encoder.transform([bundle.records[disease] for disease in gallery_ids])
    engine = SimilarityEngine(gallery_matrices, gallery_ids)

    config = json.loads(Path(args.selected_config).read_text(encoding="utf-8"))
    training = build_training_set(
        engine,
        list(relations_by_name.values()),
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
    fusion = PatternFusion(training, base)
    extractor = PaperExtractor(bundle.knowledge)

    all_rows = random_rows(
        query_ids, gallery_ids, curated_gold, paper_gold, seed=int(config["seed"])
    )
    for variant in VARIANTS:
        masked = variant != "natural_extracted"
        structured = variant != "masked_text_only"
        for level, _ in LEVELS:
            records: list[dict[str, Any]] = []
            words: list[int] = []
            counts: list[dict[str, int]] = []
            for paper in papers:
                text = paper_text(paper, level, masked=masked)
                record, extracted = extractor.extract(text, structured=structured)
                records.append(record)
                words.append(len(WORD.findall(text)))
                counts.append(extracted)
            all_rows.extend(
                score_records(
                    records,
                    query_ids,
                    encoder,
                    engine,
                    gallery_ids,
                    fusion,
                    curated_gold,
                    paper_gold,
                    condition="paper",
                    level=level,
                    variant=variant,
                    words=words,
                    extraction_counts=counts,
                )
            )
            print(f"Scored {variant}/{level}", flush=True)

    oracle_records = [copy_without_identity(bundle.records[target]) for target in query_ids]
    empty_counts = [
        {key: 0 for key in ("genes", "phenotypes", "drugs", "inheritance", "onset", "prevalence")}
        for _ in query_ids
    ]
    all_rows.extend(
        score_records(
            oracle_records,
            query_ids,
            encoder,
            engine,
            gallery_ids,
            fusion,
            curated_gold,
            paper_gold,
            condition="oracle",
            level="database_profile",
            variant="oracle_profile",
            words=[0] * len(query_ids),
            extraction_counts=empty_counts,
        )
    )

    manifest = []
    for paper in papers:
        manifest.append(
            {
                "target_id": paper.target_id,
                "target_name": bundle.records[paper.target_id]["name"],
                "stable_split": bundle.meta["split"][paper.target_id],
                "paper_id": paper.paper_id,
                "pmid": paper.pmid,
                "pmcid": paper.pmcid,
                "doi": paper.doi,
                "title": paper.title,
                "year": paper.year,
                "source_urls_json": json.dumps(paper.source_urls, ensure_ascii=False),
                "full_text_path": paper.full_text_path,
                "title_mention": int(paper.title_mention),
                "metadata_target_count": paper.metadata_target_count,
                "body_chars": paper.body_chars,
                "curated_gold_count": len(curated_gold[paper.target_id]),
                "paper_gold_count": len(paper_gold[paper.target_id]),
                "paper_gold_ids": "|".join(sorted(paper_gold[paper.target_id])),
            }
        )

    summaries = summarize(all_rows)
    generated_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "generated_at": generated_at,
        "protocol": "v2/PAPER_INPUT_BENCHMARK.md",
        "database": str(Path(args.database).resolve()),
        "query_count": len(query_ids),
        "unique_paper_count": len({paper.paper_id for paper in papers}),
        "gallery_count": len(gallery_ids),
        "train_count": len(train_ids),
        "paper_gold_queries": sum(bool(value) for value in paper_gold.values()),
        "test_candidates_with_curated_relations": len(test_candidates),
        "max_queries": args.max_queries,
        "query_ids": query_ids,
        "levels": [{"name": name, "word_limit": limit} for name, limit in LEVELS],
        "variants": list(VARIANTS),
        "modalities": list(MODALITIES),
        "selected_config": config,
        "holdout_verified": bool(
            not query_set & train_ids
            and not query_set & set(gallery_ids)
            and len({paper.paper_id for paper in papers}) == len(papers)
            and all(bundle.meta["split"][target] == "test" for target in query_ids)
        ),
    }
    write_csv(output_dir / "query_manifest.csv", manifest)
    write_csv(output_dir / "per_query.csv", all_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(
            json_safe({"metadata": metadata, "conditions": summaries}),
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    write_report(output_dir, manifest, summaries, metadata)
    print(f"Wrote benchmark artifacts to {output_dir}", flush=True)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DATABASE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--selected-config", type=Path, default=SELECTED_CONFIG)
    parser.add_argument("--max-queries", type=int, default=200)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
