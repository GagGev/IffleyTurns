"""Deterministic, token-conscious passage selection for paper extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence


WORD = re.compile(r"\S+")
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
HIGH_VALUE_TERMS = {
    "phenotype": re.compile(
        r"\b(?:phenotyp|symptom|clinical|manifest|feature|complication|"
        r"abnormalit|patient|organ|diagnos)\w*\b",
        re.I,
    ),
    "gene": re.compile(
        r"\b(?:gene|variant|mutation|allele|genetic|genomic|chromosom|"
        r"inherit|dominant|recessive|de novo|pathogenic)\w*\b",
        re.I,
    ),
    "mechanism": re.compile(
        r"\b(?:pathway|mechanism|protein|enzyme|cellular|molecular|"
        r"pathophysiolog|signaling|signalling|metaboli)\w*\b",
        re.I,
    ),
    "history": re.compile(
        r"\b(?:onset|progress|prognos|survival|severity|natural history|"
        r"infant|childhood|adult|prenatal|neonatal)\w*\b",
        re.I,
    ),
    "treatment": re.compile(
        r"\b(?:treat|therap|drug|medication|response|trial|intervention|"
        r"management|transplant)\w*\b",
        re.I,
    ),
    "prevalence": re.compile(
        r"\b(?:prevalence|incidence|epidemiolog|1\s+(?:in|per)|per\s+100[, ]?000)\b",
        re.I,
    ),
}


@dataclass(frozen=True)
class SourceUnit:
    section: str
    locator: str
    text: str


@dataclass(frozen=True)
class Passage:
    locator: str
    section: str
    text: str
    word_count: int
    relevance_score: float


def word_count(text: str) -> int:
    return len(WORD.findall(text))


def truncate_words(text: str, limit: int) -> str:
    matches = list(WORD.finditer(text))
    if len(matches) <= limit:
        return text.strip()
    return text[: matches[limit - 1].end()].strip()


def units_from_text(text: str, max_chunk_words: int = 220) -> list[SourceUnit]:
    """Split plain text into stable paragraph-like units."""

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    if not paragraphs and text.strip():
        paragraphs = [text.strip()]
    units: list[SourceUnit] = []
    for paragraph_index, paragraph in enumerate(paragraphs, 1):
        section = "Title" if paragraph_index == 1 else "Paper"
        if paragraph_index == 2:
            section = "Abstract"
        sentences = SENTENCE.split(paragraph)
        chunks: list[str] = []
        current: list[str] = []
        current_words = 0
        for sentence in sentences:
            count = word_count(sentence)
            if current and current_words + count > max_chunk_words:
                chunks.append(" ".join(current))
                current, current_words = [], 0
            if count > max_chunk_words:
                if current:
                    chunks.append(" ".join(current))
                    current, current_words = [], 0
                words = sentence.split()
                chunks.extend(
                    " ".join(words[offset : offset + max_chunk_words])
                    for offset in range(0, len(words), max_chunk_words)
                )
            else:
                current.append(sentence)
                current_words += count
        if current:
            chunks.append(" ".join(current))
        for chunk_index, chunk in enumerate(chunks or [paragraph], 1):
            units.append(
                SourceUnit(
                    section=section,
                    locator=f"paragraph:{paragraph_index}:chunk:{chunk_index}",
                    text=chunk.strip(),
                )
            )
    return units


def _score(unit: SourceUnit, index: int) -> float:
    section = unit.section.casefold()
    score = 0.0
    if "title" in section:
        score += 100.0
    if "abstract" in section:
        score += 80.0
    if index < 3:
        score += 20.0 - index
    for pattern in HIGH_VALUE_TERMS.values():
        score += min(len(pattern.findall(unit.text)), 5) * 3.0
    if re.search(r"\b(?:results?|case presentation|clinical findings?|discussion)\b", section):
        score += 8.0
    if re.search(r"\b(?:references?|bibliography|acknowledg)\b", section):
        score -= 100.0
    return score


def select_passages(
    units: Sequence[SourceUnit] | Iterable[SourceUnit],
    max_words: int = 2_000,
) -> list[Passage]:
    """Select high-value units under a hard word budget, preserving source order."""

    source = [unit for unit in units if unit.text.strip()]
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    ranked = sorted(
        enumerate(source),
        key=lambda item: (-_score(item[1], item[0]), item[0]),
    )
    chosen: dict[int, tuple[str, float]] = {}
    remaining = max_words
    for index, unit in ranked:
        if remaining <= 0:
            break
        count = word_count(unit.text)
        if count == 0:
            continue
        text = unit.text if count <= remaining else truncate_words(unit.text, remaining)
        chosen[index] = (text, _score(unit, index))
        remaining -= word_count(text)
    passages = []
    for output_index, source_index in enumerate(sorted(chosen), 1):
        text, score = chosen[source_index]
        unit = source[source_index]
        passages.append(
            Passage(
                locator=f"P{output_index:04d}",
                section=unit.section,
                text=text,
                word_count=word_count(text),
                relevance_score=score,
            )
        )
    return passages


def format_passages(passages: Sequence[Passage]) -> str:
    return "\n\n".join(
        f"[{passage.locator} | {passage.section}]\n{passage.text}"
        for passage in passages
    )

