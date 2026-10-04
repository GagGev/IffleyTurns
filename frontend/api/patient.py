"""MedGemma-backed helpers for the patient view.

MedGemma is used only for language: turning a patient's own words into
candidate clinical terms, and rewriting medical text in plain language.  It
never decides which conditions match; that is v2's placement model, and every
term it suggests is checked against the HPO vocabulary and shown to the
patient to confirm before it is used.

MedGemma is reached through any OpenAI-compatible endpoint on this machine
(medgemma_server.py, llama-server, vLLM), set with MEDGEMMA_BASE_URL.
"""

from __future__ import annotations

import html
import json
import os
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Optional
from urllib.parse import urlparse

DEFAULT_BASE_URL = "http://127.0.0.1:8766/v1"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}

INTERPRET_PROMPT = (
    "You convert a patient's own description of their symptoms into standard clinical terms, as used in the Human "
    "Phenotype Ontology. Reply with JSON only: {\"terms\": [\"...\"]}. Use singular clinical terms (for example "
    "\"Seizure\", \"Hypotonia\", \"Joint hypermobility\"). List every symptom and physical feature the person mentions, including body features such as height, finger "
    "length or delayed milestones, and nothing they do not mention. Never make "
    "a description more specific than the person said: \"a problem with a heart artery\" is \"Abnormality of the aorta\" "
    "or \"Abnormal heart morphology\", not a named complication. Do not name diseases or diagnoses. Never invent "
    "symptoms: if the text describes none, reply {\"terms\": []}.\n"
    "Examples:\n"
    "\"My daughter has fits, is very floppy and is short for her age\" -> "
    "{\"terms\": [\"Seizure\", \"Hypotonia\", \"Short stature\"]}\n"
    "\"Hello, what does this website do?\" -> {\"terms\": []}\n"
    "\"Write me a poem\" -> {\"terms\": []}"
)
SIMPLIFY_PROMPT = (
    "Rewrite medical text for a patient or family member with no medical training. Use 2 to 4 short sentences and "
    "everyday words, as if for a 12-year-old reader. Use only facts that are in the text; do not add any. Do not give a "
    "diagnosis, treatment advice or a prognosis. Reply with the rewritten text only."
)
ALTERNATIVES_PROMPT = (
    "For each clinical term below, give up to 3 official Human Phenotype Ontology term names that mean the same thing "
    "(for example \"Cloudy eyes\" -> \"Corneal opacity\", \"Delayed sitting\" -> \"Delayed ability to sit\"). Keep the "
    "same level of detail: never replace a general term with a more specific finding, complication or disease "
    "(\"Abnormality of the aorta\" -> \"Abnormal aortic morphology\", never \"Aortic aneurysm\"). Reply "
    "with JSON only, mapping each term to a list: {\"Cloudy eyes\": [\"Corneal opacity\", \"Cataract\"]}."
)
GROUP_PROMPT = (
    "You explain a group of rare conditions to a patient or family member with no medical training, in 2 to 3 short "
    "sentences of everyday words. Describe what the conditions in the group generally have in common, based only on the "
    "information given. Do not say or suggest that the reader has any of these conditions, and do not give treatment "
    "advice. Reply with the explanation only."
)


class MedGemmaUnavailable(RuntimeError):
    pass


class MedGemma:
    """Minimal client for a local OpenAI-compatible chat endpoint."""

    def __init__(self, base_url: Optional[str] = None, timeout: float = 120.0):
        self.base_url = (base_url or os.environ.get("MEDGEMMA_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        if urlparse(self.base_url).hostname not in LOCAL_HOSTS:
            # Patients' symptoms must not leave the machine.
            raise ValueError(f"MEDGEMMA_BASE_URL must point to this machine, not {self.base_url}")
        self.timeout = timeout

    def available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/models", timeout=2) as response:
                return response.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def chat(self, system: str, user: str, max_tokens: int = 300) -> str:
        body = json.dumps({
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                document = json.load(response)
        except (urllib.error.URLError, OSError) as error:
            raise MedGemmaUnavailable(
                "MedGemma is not running. Start it with `python frontend/api/medgemma_server.py --model <path>`."
            ) from error
        return str(document["choices"][0]["message"]["content"]).strip()


STOP_WORDS = {"of", "the", "a", "an", "and", "with", "to", "in", "on", "my", "his", "her", "very", "too"}


def stem(word: str) -> str:
    """Crude stem so "sitting" ~ "sit" and "delayed" ~ "delay": enough for matching against labels."""

    for suffix in ("ality", "ity", "ing", "ed", "es", "s", "y", "a", "e"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            word = word[: -len(suffix)] + ("al" if suffix == "ality" else "")
            break
    if len(word) > 3 and word[-1] == word[-2]:
        word = word[:-1]
    return word


def word_match(term: str, vocabulary: list[tuple[str, str]], limit: int) -> list[dict[str, str]]:
    """Labels containing every content word of ``term`` (stemmed, any order), shortest first."""

    stems = [stem(w) for w in re.findall(r"[a-z]+", term.lower()) if w not in STOP_WORDS]
    if not stems:
        return []
    hits = []
    for term_id, label in vocabulary:
        words = re.findall(r"[a-z]+", label.lower())
        if all(any(w.startswith(s) for w in words) for s in stems):
            hits.append((len(label), term_id, label))
    return [{"id": i, "label": l} for _, i, l in sorted(hits)[:limit]]


NEGATIONS = re.compile(r"\b(lack|lacking|absent|absence|decreased|reduced|loss|low|diminished|no|non|without|hypo\w*|aplasia)\b", re.I)


def flips_meaning(term: str, label: str) -> bool:
    """True when a label negates or reverses a term that has no such word ("Skin hyperelasticity" vs "Lack of skin elasticity")."""

    added = set(m.lower() for m in NEGATIONS.findall(label)) - set(m.lower() for m in NEGATIONS.findall(term))
    return bool(added)


def parse_alternatives(content: str) -> dict[str, list[str]]:
    match = re.search(r"\{.*\}", content, re.S)
    if not match:
        return {}
    try:
        document = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return {str(k): [str(x) for x in v] for k, v in document.items() if isinstance(v, list)}


def parse_terms(content: str) -> list[str]:
    """Pull the term list out of a reply that may be wrapped in a ```json block or prose."""

    match = re.search(r"\{.*\}", content, re.S)
    if match:
        try:
            terms = json.loads(match.group(0)).get("terms", [])
            return [str(t).strip() for t in terms if str(t).strip()]
        except (json.JSONDecodeError, AttributeError):
            pass
    return [t.strip(" -*•\"'") for t in re.split(r"[\n,;]", content) if t.strip(" -*•\"'")][:12]


class PatientHelper:
    """Language tasks for the patient view, with answers cached in memory."""

    def __init__(
        self,
        medgemma: MedGemma,
        suggest: Callable[[str, str, int], list[dict[str, str]]],
        description: Callable[[str], Optional[str]],
        vocabulary: Callable[[str], list[tuple[str, str]]] = lambda field: [],
        papers: Optional["PaperFinder"] = None,
    ):
        self.medgemma = medgemma
        self.suggest = suggest
        self.description = description
        self.vocabulary = vocabulary
        self.papers = papers
        self._cache: dict[tuple, Any] = {}
        self._lock = threading.Lock()

    def _cached(self, key: tuple, compute: Callable[[], Any]) -> Any:
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        value = compute()
        with self._lock:
            self._cache[key] = value
        return value

    def status(self) -> dict[str, Any]:
        return {"medgemma": "ready" if self.medgemma.available() else "offline"}

    def interpret(self, text: str) -> dict[str, Any]:
        """Patient's words -> clinical terms -> HPO matches for the patient to confirm."""

        text = text.strip()
        if not text:
            raise ValueError("Describe the symptoms first.")
        if len(text) > 2000:
            raise ValueError("Please keep the description under 2,000 characters.")
        terms = self._cached(("interpret", text), lambda: parse_terms(self.medgemma.chat(INTERPRET_PROMPT, text, 200)))
        results = [{"term": term, "matches": self._match(term)} for term in terms]
        unmatched = [r["term"] for r in results if not r["matches"]]
        if unmatched:
            # Ask once for official names of everything that did not match, then look those up.
            alternatives = self._cached(
                ("alternatives", tuple(unmatched)),
                lambda: parse_alternatives(self.medgemma.chat(ALTERNATIVES_PROMPT, "\n".join(unmatched), 250)),
            )
            for r in results:
                for alternative in alternatives.get(r["term"], []) if not r["matches"] else []:
                    r["matches"] = self._match(alternative)
                    if r["matches"]:
                        break
        return {"terms": results}

    def _match(self, term: str) -> list[dict[str, str]]:
        """HPO terms for one clinical term: exact label first, then prefix/word matches, then all words in any order."""

        matches = self.suggest("phenotypes", term, 4)
        if not matches and term.lower().endswith("s"):
            matches = self.suggest("phenotypes", term[:-1], 4)
        if not matches:
            matches = word_match(term, self.vocabulary("phenotypes"), 4)
        matches = [m for m in matches if not flips_meaning(term, m["label"])]
        exact = [m for m in matches if m["label"].lower() == term.lower()]
        return exact + [m for m in matches if m not in exact]

    def explain_disease(self, disease_id: str, name: str) -> dict[str, Optional[str]]:
        """Plain-language version of the Orphanet description; ``text`` is None when Orphanet has none (~16%)."""

        if not disease_id:
            raise ValueError("Send the disease's ID.")
        source = self.description(disease_id)
        if not source:
            return {"id": disease_id, "text": None}
        text = self._cached(("disease", disease_id), lambda: self.medgemma.chat(SIMPLIFY_PROMPT, source[:3000], 220))
        return {"id": disease_id, "text": text}

    def key_papers(self, disease_id: str) -> dict[str, Any]:
        if not disease_id:
            raise ValueError("Send the disease's ID.")
        if self.papers is None:
            raise RuntimeError("Paper lookup is not configured.")
        return self.papers.papers(disease_id)

    def explain_group(self, group: dict[str, Any]) -> dict[str, str]:
        category = str(group.get("category", ""))
        examples = [str(x) for x in group.get("examples", [])][:6]
        shared = [str(x) for x in group.get("shared", [])][:8]
        if not category and not examples:
            raise ValueError("Send the group's category and example conditions.")
        user = (
            f"Group: {category}\n"
            f"Example conditions in the group: {', '.join(examples)}\n"
            f"Features these conditions often share: {', '.join(shared) or 'not listed'}"
        )
        text = self._cached(("group", category, tuple(examples), tuple(shared)), lambda: self.medgemma.chat(GROUP_PROMPT, user, 180))
        return {"text": text}


# --- Key papers ------------------------------------------------------------------------

EUROPE_PMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
GENERIC_WORDS = {
    "syndrome", "disease", "disorder", "familial", "isolated", "congenital", "type", "related", "deficiency",
    "hereditary", "inherited", "form", "rare", "primary", "secondary", "autosomal", "dominant", "recessive", "linked",
    "early", "onset", "adult", "juvenile", "infantile", "neonatal", "childhood", "with", "without", "and", "due",
}
MAX_SYNONYM_HITS = 200  # A synonym matching more papers than this is a broad term, not this disease.


ANIMAL_STUDY = re.compile(
    r"\b(dogs?|cats?|canine|feline|bovine|equine|porcine|ovine|horses?|cattle|mice|mouse|murine|rats?|zebrafish|sheep|"
    r"pigs?|calf|calves|primates?|veterinary)\b",
    re.I,
)


def distinctive_words(name: str) -> set[str]:
    """Stemmed content words, so "dermoids" in a title matches "dermoid" in a disease name."""

    return {stem(w) for w in re.findall(r"[a-z0-9]+", name.lower()) if len(w) >= 4 and w not in GENERIC_WORDS}


def europe_pmc_search(query: str, page_size: int = 6) -> tuple[int, list[dict[str, Any]]]:
    params = {"query": f"{query} AND SRC:MED", "format": "json", "pageSize": page_size, "sort": "CITED desc", "resultType": "lite"}
    url = EUROPE_PMC + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=12) as response:
        document = json.load(response)
    return int(document.get("hitCount") or 0), document.get("resultList", {}).get("result", [])


class PaperFinder:
    """The most-cited papers about a disease from Europe PMC, by its name or a specific synonym.

    Only the public disease name is sent.  A paper is kept only if its title shares a distinctive word with the name
    searched, so a synonym that happens to appear in an unrelated abstract does not produce a wrong paper.
    """

    def __init__(
        self,
        record: Callable[[str], Optional[dict[str, Any]]],
        search: Callable[[str], tuple[int, list[dict[str, Any]]]] = europe_pmc_search,
        limit: int = 2,
    ):
        self.record = record
        self.search = search
        self.limit = limit
        self._cache: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def papers(self, disease_id: str) -> dict[str, Any]:
        with self._lock:
            if disease_id in self._cache:
                return self._cache[disease_id]
        record = self.record(disease_id)
        if not record:
            raise ValueError(f"Unknown disease {disease_id!r}.")
        name = str(record.get("name") or disease_id)
        search_url = "https://europepmc.org/search?" + urllib.parse.urlencode({"query": f'"{name}"'})
        try:
            papers = self._find(name, [str(s) for s in record.get("synonyms") or []])
        except (urllib.error.URLError, OSError, ValueError):
            return {"id": disease_id, "papers": [], "searchUrl": search_url, "error": "Europe PMC could not be reached."}
        result = {"id": disease_id, "papers": papers, "searchUrl": search_url}
        with self._lock:
            self._cache[disease_id] = result
        return result

    def _find(self, name: str, synonyms: list[str]) -> list[dict[str, Any]]:
        specific = [s for s in synonyms if len(s) > 5 and not s.isupper() and s.lower() != name.lower()]
        attempts = [("TITLE", name, False)] + [("TITLE", s, True) for s in specific]
        attempts += [("TITLE_ABS", name, False)] + [("TITLE_ABS", s, True) for s in specific]
        found: list[dict[str, Any]] = []
        for field, term, is_synonym in attempts:
            words = distinctive_words(term) or distinctive_words(name)
            if not words:
                continue
            hits, results = self.search(f'{field}:"{term}"')
            if is_synonym and hits > MAX_SYNONYM_HITS:
                continue
            for r in results:
                title = re.sub(r"<[^>]+>", "", html.unescape(str(r.get("title") or ""))).strip()
                # The title must share two distinctive words with the term (one if it has only one), and be about people.
                if not title or len(distinctive_words(title) & words) < min(2, len(words)) or ANIMAL_STUDY.search(title):
                    continue
                if any(p["title"] == title for p in found):
                    continue
                pmcid = r.get("pmcid")
                url = (
                    f"https://europepmc.org/article/PMC/{pmcid}" if pmcid
                    else f"https://europepmc.org/article/MED/{r.get('pmid')}" if r.get("pmid")
                    else f"https://doi.org/{r.get('doi')}" if r.get("doi") else None
                )
                if not url:
                    continue
                found.append({
                    "title": title,
                    "year": r.get("pubYear"),
                    "journal": r.get("journalTitle"),
                    "citedBy": int(r.get("citedByCount") or 0),
                    "openAccess": r.get("isOpenAccess") == "Y",
                    "url": url,
                })
                if len(found) >= self.limit:
                    return found
        return found
