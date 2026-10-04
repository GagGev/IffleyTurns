"""Resumable rare-disease relationship acquisition from Europe PMC and PubMed.

The collector uses only documented APIs and local Orphanet data.  Exact-name
matching retrieves candidate passages before a conservative rule-based scoring
pass.  Every generated score is explicitly unverified and queued for review.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
import random
import re
import sqlite3
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / ".data/literature_acquisition/papers.sqlite"
ORPHA_XML = ROOT / ".data/databases/orphadata/en_product1.xml"
EUROPE_PMC_API = "https://www.ebi.ac.uk/europepmc/webservices/rest"
PUBMED_API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PIPELINE_VERSION = "rare-disease-relationships-1.0"
PUBMED_MAX_IDS_PER_SEARCH = 9999

DIMENSIONS = (
    "clinical_phenotype",
    "genetic_etiology",
    "molecular_mechanism",
    "natural_history",
    "therapeutic_similarity",
)

DIMENSION_PATTERNS = {
    "clinical_phenotype": re.compile(
        r"\b(phenotyp\w*|symptom\w*|clinical (?:feature|presentation|manifestation)s?|"
        r"manifestation\w*|organ systems?|complication\w*|signs?|mimic\w*|"
        r"differential diagnos\w*|affected organs?)\b",
        re.I,
    ),
    "genetic_etiology": re.compile(
        r"\b(gene\w*|genetic\w*|genotyp\w*|variant\w*|mutation\w*|locus|loci|"
        r"chromosom\w*|inherit\w*|allel\w*|de novo|autosomal|x-linked)\b",
        re.I,
    ),
    "molecular_mechanism": re.compile(
        r"\b(pathway\w*|protein\w*|molecular\w*|cellular\w*|mechanis\w*|"
        r"pathogenes\w*|pathophysiolog\w*|enzyme\w*|receptor\w*|"
        r"signalling|signaling|biological process\w*)\b",
        re.I,
    ),
    "natural_history": re.compile(
        r"\b(natural history|age (?:at|of) onset|onset|progress\w*|disease course|"
        r"sever\w*|prognos\w*|survival|mortality|life expectancy|outcome\w*)\b",
        re.I,
    ),
    "therapeutic_similarity": re.compile(
        r"\b(treat\w*|therap\w*|drug\w*|medication\w*|intervention\w*|"
        r"therapeutic target\w*|treatment response|respond\w* to|transplant\w*)\b",
        re.I,
    ),
}

RELATION_PATTERN = re.compile(
    r"\b(shared?|similar\w*|overlap\w*|both|same|common|compar\w*|"
    r"distinguish\w*|differ\w*|distinct|unrelated|contrast\w*|mimic\w*|"
    r"associated?|relationship\w*|spectrum|co-occur\w*|caus\w*|"
    r"responsible for|linked?|resembl\w*)\b",
    re.I,
)

SCORE_PATTERNS = (
    (
        0,
        "explicit_non_overlap",
        re.compile(
            r"\b(no (?:meaningful )?(?:shared|common|overlap)|without (?:shared|common)|"
            r"not associated|unrelated|fundamentally different|distinct "
            r"(?:phenotyp|genetic|molecular|mechanis|pathway|course|treat)|"
            r"(?:phenotyp|mechanis|pathway|course|treat)\w*.{0,45}\bdiffer\w*)\b",
            re.I,
        ),
        5,
    ),
    (
        4,
        "very_strong_equivalence",
        re.compile(
            r"\b(indistinguishable|near[- ]identical|essentially (?:the )?same|"
            r"phenotypically identical|same (?:principal )?(?:causal )?"
            r"(?:gene|variant|mutation|molecular mechanism|pathway)|"
            r"same disease[- ]modifying (?:treatment|therapy)|near[- ]equivalent)\b",
            re.I,
        ),
        5,
    ),
    (
        3,
        "strong_shared_evidence",
        re.compile(
            r"\b(strong(?:ly)? similar|substantial(?:ly)? (?:shared|overlap)|"
            r"extensive(?:ly)? (?:shared|overlap)|closely related|clinical mimic|"
            r"shared (?:central |causal |core |key )"
            r"(?:pathway|mechanism|gene|variant|phenotype|treatment)|"
            r"common (?:causal|pathogenic) (?:gene|variant|mutation))\b",
            re.I,
        ),
        4,
    ),
    (
        2,
        "moderate_overlap",
        re.compile(
            r"\b(partial(?:ly)? (?:shared|overlap|similar)|overlap\w*|similar\w*|"
            r"shared|both|in common|common (?:feature|symptom|pathway|treatment|gene))\b",
            re.I,
        ),
        3,
    ),
    (
        1,
        "weak_or_broad_relationship",
        re.compile(
            r"\b(related|associated|linked|resembl\w*|compar\w*|same organ system|"
            r"broad(?:ly)? similar)\b",
            re.I,
        ),
        2,
    ),
)

ALLOWED_ENTITY_TYPES = {
    "Disease",
    "Malformation syndrome",
    "Clinical subtype",
    "Etiological subtype",
    "Clinical syndrome",
}

GENERIC_ALIASES = {
    "disease",
    "rare disease",
    "syndrome",
    "disorder",
    "malformation syndrome",
    "developmental disorder",
    "genetic disorder",
    "inherited disorder",
}

EUROPE_PMC_QUERY = (
    '(TITLE_ABS:"rare disease" OR TITLE_ABS:"rare diseases" OR '
    'TITLE_ABS:"rare disorder" OR TITLE_ABS:"rare disorders" OR '
    'TITLE_ABS:"orphan disease" OR TITLE_ABS:"orphan diseases") AND '
    "(TITLE_ABS:similar* OR TITLE_ABS:shared OR TITLE_ABS:overlap* OR "
    'TITLE_ABS:"differential diagnosis" OR TITLE_ABS:mimic* OR '
    "TITLE_ABS:distinguish* OR TITLE_ABS:compar* OR TITLE_ABS:relationship* OR "
    "TITLE_ABS:association* OR TITLE_ABS:mechanism*)"
)

PUBMED_QUERY = (
    '(("Rare Diseases"[Mesh] OR "rare disease"[Title/Abstract] OR '
    '"rare diseases"[Title/Abstract] OR "rare disorder"[Title/Abstract] OR '
    '"rare disorders"[Title/Abstract] OR "orphan disease"[Title/Abstract]) AND '
    "(similar*[Title/Abstract] OR shared[Title/Abstract] OR "
    'overlap*[Title/Abstract] OR "differential diagnosis"[Title/Abstract] OR '
    "mimic*[Title/Abstract] OR distinguish*[Title/Abstract] OR "
    "compar*[Title/Abstract] OR relationship*[Title/Abstract] OR "
    "association*[Title/Abstract] OR mechanism*[Title/Abstract]))"
)


SCHEMA = """
PRAGMA foreign_keys=ON;
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS papers(
    paper_id INTEGER PRIMARY KEY,
    pmid TEXT,
    pmcid TEXT,
    doi TEXT,
    normalized_title TEXT NOT NULL,
    title TEXT NOT NULL,
    abstract TEXT,
    year INTEGER,
    journal TEXT,
    publication_types_json TEXT NOT NULL DEFAULT '[]',
    sources_json TEXT NOT NULL DEFAULT '[]',
    source_urls_json TEXT NOT NULL DEFAULT '[]',
    is_open_access INTEGER NOT NULL DEFAULT 0,
    access_status TEXT NOT NULL DEFAULT 'metadata_only'
        CHECK(access_status IN ('metadata_only','abstract_only','open_full_text','unavailable')),
    full_text_raw_path TEXT,
    license TEXT,
    discovered_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    provenance_json TEXT NOT NULL DEFAULT '{}'
);

CREATE UNIQUE INDEX IF NOT EXISTS papers_pmid_uq ON papers(pmid) WHERE pmid IS NOT NULL AND pmid!='';
CREATE UNIQUE INDEX IF NOT EXISTS papers_pmcid_uq ON papers(pmcid) WHERE pmcid IS NOT NULL AND pmcid!='';
CREATE UNIQUE INDEX IF NOT EXISTS papers_doi_uq ON papers(doi) WHERE doi IS NOT NULL AND doi!='';
CREATE UNIQUE INDEX IF NOT EXISTS papers_title_uq ON papers(normalized_title) WHERE normalized_title!='';

CREATE TABLE IF NOT EXISTS orpha_entities(
    orpha_id TEXT PRIMARY KEY,
    preferred_name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    source_release TEXT,
    source_sha256 TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS orpha_aliases(
    normalized_alias TEXT PRIMARY KEY,
    display_alias TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    candidate_orpha_ids_json TEXT NOT NULL,
    mapping_status TEXT NOT NULL CHECK(mapping_status IN ('exact','ambiguous'))
);

CREATE TABLE IF NOT EXISTS fulltext_sections(
    section_id INTEGER PRIMARY KEY,
    paper_id INTEGER NOT NULL REFERENCES papers ON DELETE CASCADE,
    section_name TEXT NOT NULL,
    source_locator TEXT NOT NULL,
    text TEXT NOT NULL,
    UNIQUE(paper_id,source_locator)
);

CREATE TABLE IF NOT EXISTS disease_mentions(
    mention_id INTEGER PRIMARY KEY,
    paper_id INTEGER NOT NULL REFERENCES papers ON DELETE CASCADE,
    orpha_id TEXT REFERENCES orpha_entities,
    mention_text TEXT NOT NULL,
    normalized_term TEXT NOT NULL,
    mapping_status TEXT NOT NULL CHECK(mapping_status IN ('exact','ambiguous')),
    candidate_orpha_ids_json TEXT NOT NULL,
    section_name TEXT NOT NULL,
    source_locator TEXT NOT NULL,
    source_status TEXT NOT NULL CHECK(source_status IN ('abstract_only','full_text')),
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    UNIQUE(paper_id,normalized_term,source_locator,char_start,char_end)
);

CREATE TABLE IF NOT EXISTS paper_pairs(
    paper_pair_id INTEGER PRIMARY KEY,
    paper_id INTEGER NOT NULL REFERENCES papers ON DELETE CASCADE,
    orpha_id_a TEXT NOT NULL REFERENCES orpha_entities,
    orpha_id_b TEXT NOT NULL REFERENCES orpha_entities,
    pair_key TEXT NOT NULL,
    evidence_status TEXT NOT NULL DEFAULT 'comparative_evidence_detected',
    best_access_status TEXT NOT NULL CHECK(best_access_status IN ('abstract_only','full_text')),
    created_at TEXT NOT NULL,
    CHECK(orpha_id_a < orpha_id_b),
    UNIQUE(paper_id,pair_key)
);

CREATE TABLE IF NOT EXISTS dimension_scores(
    score_id INTEGER PRIMARY KEY,
    paper_pair_id INTEGER NOT NULL REFERENCES paper_pairs ON DELETE CASCADE,
    dimension TEXT NOT NULL,
    ordinal_score INTEGER CHECK(ordinal_score BETWEEN 0 AND 4),
    score_label TEXT NOT NULL,
    supporting_passage_or_paraphrase TEXT,
    source_section TEXT,
    source_locator TEXT,
    evidence_access_status TEXT NOT NULL
        CHECK(evidence_access_status IN ('abstract_only','full_text','not_assessed')),
    extraction_confidence INTEGER NOT NULL CHECK(extraction_confidence BETWEEN 1 AND 3),
    extraction_method TEXT NOT NULL DEFAULT 'automated',
    verification_status TEXT NOT NULL DEFAULT 'unverified'
        CHECK(verification_status IN ('unverified','human_verified')),
    scoring_rule TEXT NOT NULL,
    created_at TEXT NOT NULL,
    CHECK(
        (ordinal_score IS NULL AND score_label='NA') OR
        (ordinal_score IS NOT NULL AND score_label IN ('0','1','2','3','4')
         AND supporting_passage_or_paraphrase IS NOT NULL
         AND source_locator IS NOT NULL)
    ),
    CHECK(NOT (extraction_method='automated' AND verification_status='human_verified')),
    UNIQUE(paper_pair_id,dimension)
);

CREATE TABLE IF NOT EXISTS search_log(
    request_key TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    request_kind TEXT NOT NULL,
    query_text TEXT,
    request_url TEXT NOT NULL,
    page_token TEXT,
    requested_at TEXT NOT NULL,
    completed_at TEXT,
    status_code INTEGER,
    result_count INTEGER,
    records_received INTEGER,
    response_sha256 TEXT,
    raw_path TEXT,
    status TEXT NOT NULL CHECK(status IN ('started','complete','failed')),
    error TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    pipeline_version TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS review_queue(
    queue_id INTEGER PRIMARY KEY,
    queue_key TEXT NOT NULL UNIQUE,
    paper_id INTEGER NOT NULL REFERENCES papers ON DELETE CASCADE,
    paper_pair_id INTEGER REFERENCES paper_pairs ON DELETE CASCADE,
    score_id INTEGER REFERENCES dimension_scores ON DELETE CASCADE,
    reason TEXT NOT NULL,
    priority INTEGER NOT NULL CHECK(priority BETWEEN 1 AND 3),
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK(status IN ('pending','in_review','verified','rejected')),
    details_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rejected_records(
    paper_id INTEGER PRIMARY KEY REFERENCES papers ON DELETE CASCADE,
    reason TEXT NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pipeline_state(
    state_key TEXT PRIMARY KEY,
    state_value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def normalize_space(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_doi(value: str | None) -> str:
    value = normalize_space(value).casefold()
    value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value)
    return value.removeprefix("doi:").strip().rstrip(".")


def normalize_name(value: str | None) -> str:
    value = unicodedata.normalize("NFKD", value or "").casefold()
    value = "".join(c for c in value if not unicodedata.combining(c))
    return " ".join(re.findall(r"[a-z0-9]+", value))


def normalized_title(value: str | None) -> str:
    value = normalize_name(value)
    return re.sub(r"^(?:the|a|an) ", "", value)


def element_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return normalize_space("".join(element.itertext()))


def connect(db_path: Path = DEFAULT_DB) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA busy_timeout=30000")
    db.executescript(SCHEMA)
    return db


def set_state(db: sqlite3.Connection, key: str, value: object) -> None:
    db.execute(
        """INSERT INTO pipeline_state(state_key,state_value,updated_at) VALUES(?,?,?)
        ON CONFLICT(state_key) DO UPDATE SET state_value=excluded.state_value,
        updated_at=excluded.updated_at""",
        (key, json.dumps(value, ensure_ascii=False), utc_now()),
    )


def get_state(db: sqlite3.Connection, key: str, default: object = None) -> object:
    row = db.execute(
        "SELECT state_value FROM pipeline_state WHERE state_key=?", (key,)
    ).fetchone()
    return json.loads(row[0]) if row else default


def import_orphanet(db: sqlite3.Connection, xml_path: Path = ORPHA_XML) -> None:
    """Import disease entities and aliases once, preserving ambiguous mappings."""
    if db.execute("SELECT COUNT(*) FROM orpha_entities").fetchone()[0]:
        return
    if not xml_path.exists():
        raise FileNotFoundError(f"Missing local Orphanet dictionary: {xml_path}")
    source_hash = sha256(xml_path.read_bytes())
    root = ET.parse(xml_path).getroot()
    aliases: dict[str, dict[str, object]] = {}
    entities = []
    for node in root.findall("DisorderList/Disorder"):
        entity_type = normalize_space(node.findtext("DisorderType/Name"))
        if entity_type not in ALLOWED_ENTITY_TYPES:
            continue
        code = normalize_space(node.findtext("OrphaCode"))
        preferred = normalize_space(node.findtext("Name"))
        if not code or not preferred:
            continue
        orpha_id = f"ORPHA:{code}"
        entities.append(
            (
                orpha_id,
                preferred,
                entity_type,
                root.attrib.get("date") or root.attrib.get("version"),
                source_hash,
            )
        )
        terms = [(preferred, "preferred")]
        terms.extend(
            (normalize_space(s.text), "synonym")
            for s in node.findall("SynonymList/Synonym")
            if normalize_space(s.text)
        )
        for display, _kind in terms:
            normalized = normalize_name(display)
            tokens = normalized.split()
            if (
                normalized in GENERIC_ALIASES
                or len(normalized) < 6
                or not tokens
                or len(tokens) > 14
                or (len(tokens) == 1 and len(tokens[0]) < 8)
                or (display.isupper() and len(display) <= 7)
            ):
                continue
            entry = aliases.setdefault(
                normalized, {"display": display, "ids": set(), "tokens": len(tokens)}
            )
            entry["ids"].add(orpha_id)
    db.executemany(
        """INSERT INTO orpha_entities(
        orpha_id,preferred_name,entity_type,source_release,source_sha256
        ) VALUES(?,?,?,?,?)""",
        entities,
    )
    db.executemany(
        """INSERT INTO orpha_aliases(
        normalized_alias,display_alias,token_count,candidate_orpha_ids_json,mapping_status
        ) VALUES(?,?,?,?,?)""",
        [
            (
                alias,
                str(data["display"]),
                int(data["tokens"]),
                json.dumps(sorted(data["ids"])),
                "exact" if len(data["ids"]) == 1 else "ambiguous",
            )
            for alias, data in aliases.items()
        ],
    )
    set_state(
        db,
        "orphanet",
        {
            "path": str(xml_path),
            "sha256": source_hash,
            "release": root.attrib,
            "entities": len(entities),
            "aliases": len(aliases),
        },
    )
    db.commit()


@dataclass(frozen=True)
class Mention:
    mention_text: str
    normalized_term: str
    candidate_ids: tuple[str, ...]
    start: int
    end: int

    @property
    def mapping_status(self) -> str:
        return "exact" if len(self.candidate_ids) == 1 else "ambiguous"

    @property
    def orpha_id(self) -> str | None:
        return self.candidate_ids[0] if len(self.candidate_ids) == 1 else None


class DiseaseMapper:
    """Exact Orphanet alias matcher using a token-prefix index."""

    def __init__(self, aliases: Iterable[tuple[str, Iterable[str]]]):
        self.by_first: dict[str, list[tuple[tuple[str, ...], str, tuple[str, ...]]]] = (
            collections.defaultdict(list)
        )
        for alias, ids in aliases:
            tokens = tuple(alias.split())
            if tokens:
                self.by_first[tokens[0]].append((tokens, alias, tuple(sorted(ids))))
        for values in self.by_first.values():
            values.sort(key=lambda item: (-len(item[0]), item[1]))

    @classmethod
    def from_db(cls, db: sqlite3.Connection) -> "DiseaseMapper":
        return cls(
            (
                row["normalized_alias"],
                json.loads(row["candidate_orpha_ids_json"]),
            )
            for row in db.execute(
                "SELECT normalized_alias,candidate_orpha_ids_json FROM orpha_aliases"
            )
        )

    def detect(self, text: str) -> list[Mention]:
        words = list(re.finditer(r"[A-Za-z0-9]+", text))
        tokens = [m.group(0).casefold() for m in words]
        found: list[Mention] = []
        i = 0
        while i < len(tokens):
            match = None
            for alias_tokens, alias, ids in self.by_first.get(tokens[i], ()):
                length = len(alias_tokens)
                if tuple(tokens[i : i + length]) == alias_tokens:
                    match = (length, alias, ids)
                    break
            if match is None:
                i += 1
                continue
            length, alias, ids = match
            start, end = words[i].start(), words[i + length - 1].end()
            found.append(Mention(text[start:end], alias, ids, start, end))
            i += length
        return found


class ApiClient:
    """Rate-limited HTTP client with an immutable on-disk response cache."""

    def __init__(self, db: sqlite3.Connection, data_dir: Path):
        self.db = db
        self.data_dir = data_dir
        self.raw_dir = data_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.last_request: dict[str, float] = collections.defaultdict(float)
        self.ncbi_api_key = os.environ.get("NCBI_API_KEY", "")
        email = os.environ.get("NCBI_EMAIL", "")
        self.user_agent = "RareDiseaseRelationshipEvidence/1.0"
        if email:
            self.user_agent += f" ({email})"

    def _delay(self, source: str) -> None:
        # NCBI permits at most 3 requests/s without a key and 10/s with one.
        interval = 0.11 if source == "PubMed" and self.ncbi_api_key else 0.35
        if source == "Europe PMC":
            interval = 0.2
        elapsed = time.monotonic() - self.last_request[source]
        time.sleep(max(0.0, interval - elapsed))
        self.last_request[source] = time.monotonic()

    def get(
        self,
        source: str,
        endpoint: str,
        params: dict[str, object] | None,
        request_kind: str,
        query_text: str | None = None,
        page_token: str | None = None,
    ) -> tuple[bytes | None, sqlite3.Row]:
        params = dict(params or {})
        if source == "PubMed":
            params.setdefault("tool", "rare_disease_relationship_evidence")
            if os.environ.get("NCBI_EMAIL"):
                params.setdefault("email", os.environ["NCBI_EMAIL"])
            if self.ncbi_api_key:
                params.setdefault("api_key", self.ncbi_api_key)
        query = urllib.parse.urlencode(sorted(params.items()), doseq=True)
        url = endpoint + (("?" + query) if query else "")
        request_key = sha256(url)
        cached = self.db.execute(
            "SELECT * FROM search_log WHERE request_key=?", (request_key,)
        ).fetchone()
        if cached and cached["status"] == "complete" and cached["raw_path"]:
            path = self.data_dir / cached["raw_path"]
            if path.exists():
                body = path.read_bytes()
                if sha256(body) == cached["response_sha256"]:
                    return body, cached
        requested = utc_now()
        self.db.execute(
            """INSERT INTO search_log(
            request_key,source,request_kind,query_text,request_url,page_token,
            requested_at,status,attempts,pipeline_version
            ) VALUES(?,?,?,?,?,?,?,'started',0,?)
            ON CONFLICT(request_key) DO UPDATE SET requested_at=excluded.requested_at,
            status='started',error=NULL,pipeline_version=excluded.pipeline_version""",
            (
                request_key,
                source,
                request_kind,
                query_text,
                url,
                page_token,
                requested,
                PIPELINE_VERSION,
            ),
        )
        self.db.commit()
        body = b""
        status_code = None
        error = None
        attempts = 0
        for attempts in range(1, 5):
            self._delay(source)
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": self.user_agent,
                    "Accept": "application/json, application/xml;q=0.9",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    body = response.read()
                    status_code = response.status
                error = None
                break
            except urllib.error.HTTPError as exc:
                status_code = exc.code
                body = exc.read()
                error = str(exc)
                if exc.code not in (429, 500, 502, 503, 504):
                    break
                retry_after = exc.headers.get("Retry-After", "")
                pause = (
                    float(retry_after)
                    if retry_after.replace(".", "", 1).isdigit()
                    else 2**attempts + random.random()
                )
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                error = repr(exc)
                pause = 2**attempts + random.random()
            if attempts < 4:
                time.sleep(pause)
        relative = Path("raw") / source.lower().replace(" ", "_") / f"{request_key}.response"
        path = self.data_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        complete = status_code == 200 and error is None
        self.db.execute(
            """UPDATE search_log SET completed_at=?,status_code=?,response_sha256=?,
            raw_path=?,status=?,error=?,attempts=? WHERE request_key=?""",
            (
                utc_now(),
                status_code,
                sha256(body),
                relative.as_posix(),
                "complete" if complete else "failed",
                error,
                attempts,
                request_key,
            ),
        )
        self.db.commit()
        row = self.db.execute(
            "SELECT * FROM search_log WHERE request_key=?", (request_key,)
        ).fetchone()
        return (body if complete else None), row

    def annotate(
        self, request_key: str, result_count: int | None, records_received: int
    ) -> None:
        self.db.execute(
            "UPDATE search_log SET result_count=?,records_received=? WHERE request_key=?",
            (result_count, records_received, request_key),
        )
        self.db.commit()


def _json_list(value: str | None) -> list[str]:
    try:
        parsed = json.loads(value or "[]")
        return list(parsed) if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def _matching_paper_ids(db: sqlite3.Connection, record: dict[str, object]) -> list[int]:
    clauses, values = [], []
    for field in ("pmid", "pmcid", "doi"):
        value = normalize_space(str(record.get(field) or ""))
        if value:
            clauses.append(f"{field}=?")
            values.append(value)
    title_key = normalized_title(str(record.get("title") or ""))
    if title_key:
        clauses.append("normalized_title=?")
        values.append(title_key)
    if not clauses:
        return []
    return [
        row[0]
        for row in db.execute(
            f"SELECT paper_id FROM papers WHERE {' OR '.join(clauses)}", values
        )
    ]


def _merge_duplicate_papers(
    db: sqlite3.Connection, paper_ids: list[int]
) -> int:
    keep = min(paper_ids)
    rows = db.execute(
        f"SELECT * FROM papers WHERE paper_id IN ({','.join('?' * len(paper_ids))})",
        paper_ids,
    ).fetchall()
    merged_sources: set[str] = set()
    merged_urls: set[str] = set()
    for row in rows:
        merged_sources.update(_json_list(row["sources_json"]))
        merged_urls.update(_json_list(row["source_urls_json"]))
    # Generated analysis can be rebuilt deterministically after a late merge.
    for duplicate in sorted(set(paper_ids) - {keep}):
        db.execute("DELETE FROM papers WHERE paper_id=?", (duplicate,))
    db.execute(
        "UPDATE papers SET sources_json=?,source_urls_json=?,updated_at=? WHERE paper_id=?",
        (
            json.dumps(sorted(merged_sources)),
            json.dumps(sorted(merged_urls)),
            utc_now(),
            keep,
        ),
    )
    return keep


def upsert_paper(
    db: sqlite3.Connection,
    record: dict[str, object],
    source: str,
    allow_insert: bool = True,
) -> int | None:
    """Deduplicate in PMID, PMCID, DOI, normalized-title order."""
    record = dict(record)
    record["pmid"] = normalize_space(str(record.get("pmid") or "")) or None
    record["pmcid"] = normalize_space(str(record.get("pmcid") or "")).upper() or None
    record["doi"] = normalize_doi(str(record.get("doi") or "")) or None
    record["title"] = normalize_space(str(record.get("title") or ""))
    title_key = normalized_title(record["title"])
    if not title_key:
        return None
    matches = _matching_paper_ids(db, record)
    if len(set(matches)) > 1:
        paper_id = _merge_duplicate_papers(db, sorted(set(matches)))
    elif matches:
        paper_id = matches[0]
    else:
        if not allow_insert:
            return None
        timestamp = utc_now()
        abstract = normalize_space(str(record.get("abstract") or ""))
        cursor = db.execute(
            """INSERT INTO papers(
            pmid,pmcid,doi,normalized_title,title,abstract,year,journal,
            publication_types_json,sources_json,source_urls_json,is_open_access,
            access_status,discovered_at,updated_at,provenance_json
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                record["pmid"],
                record["pmcid"],
                record["doi"],
                title_key,
                record["title"],
                abstract or None,
                record.get("year"),
                normalize_space(str(record.get("journal") or "")) or None,
                json.dumps(record.get("publication_types") or []),
                json.dumps([source]),
                json.dumps([str(record.get("source_url") or "")]),
                int(bool(record.get("is_open_access"))),
                "abstract_only" if abstract else "metadata_only",
                timestamp,
                timestamp,
                json.dumps(record.get("provenance") or {}, ensure_ascii=False),
            ),
        )
        return int(cursor.lastrowid)
    current = db.execute("SELECT * FROM papers WHERE paper_id=?", (paper_id,)).fetchone()
    sources = sorted(set(_json_list(current["sources_json"])) | {source})
    urls = sorted(
        set(_json_list(current["source_urls_json"]))
        | ({str(record.get("source_url"))} if record.get("source_url") else set())
    )
    abstract = current["abstract"] or normalize_space(str(record.get("abstract") or ""))
    access = (
        "open_full_text"
        if current["access_status"] == "open_full_text"
        else "abstract_only"
        if abstract
        else "metadata_only"
    )
    updates = {
        "pmid": current["pmid"] or record["pmid"],
        "pmcid": current["pmcid"] or record["pmcid"],
        "doi": current["doi"] or record["doi"],
        "title": current["title"] or record["title"],
        "abstract": abstract or None,
        "year": current["year"] or record.get("year"),
        "journal": current["journal"]
        or normalize_space(str(record.get("journal") or ""))
        or None,
        "publication_types_json": current["publication_types_json"]
        if _json_list(current["publication_types_json"])
        else json.dumps(record.get("publication_types") or []),
        "sources_json": json.dumps(sources),
        "source_urls_json": json.dumps(urls),
        "is_open_access": max(
            int(current["is_open_access"]), int(bool(record.get("is_open_access")))
        ),
        "access_status": access,
        "updated_at": utc_now(),
    }
    db.execute(
        """UPDATE papers SET pmid=:pmid,pmcid=:pmcid,doi=:doi,title=:title,
        abstract=:abstract,year=:year,journal=:journal,
        publication_types_json=:publication_types_json,sources_json=:sources_json,
        source_urls_json=:source_urls_json,is_open_access=:is_open_access,
        access_status=:access_status,updated_at=:updated_at WHERE paper_id=:paper_id""",
        dict(updates, paper_id=paper_id),
    )
    return paper_id


def epmc_record(record: dict[str, object]) -> dict[str, object]:
    publication_types = record.get("pubTypeList") or {}
    if isinstance(publication_types, dict):
        publication_types = publication_types.get("pubType") or []
    pmid = record.get("pmid")
    if not pmid and record.get("source") == "MED":
        pmid = record.get("id")
    return {
        "pmid": pmid,
        "pmcid": record.get("pmcid"),
        "doi": record.get("doi"),
        "title": record.get("title"),
        "abstract": record.get("abstractText"),
        "year": int(record.get("pubYear") or 0) or None,
        "journal": record.get("journalTitle"),
        "publication_types": publication_types,
        "is_open_access": record.get("isOpenAccess") == "Y",
        "source_url": (
            f"https://europepmc.org/article/{record.get('source')}/{record.get('id')}"
        ),
        "provenance": {
            "europe_pmc_source": record.get("source"),
            "europe_pmc_id": record.get("id"),
        },
    }


def _pubmed_year(article: ET.Element) -> int | None:
    values = [
        article.findtext("./MedlineCitation/Article/Journal/JournalIssue/PubDate/Year"),
        article.findtext("./MedlineCitation/DateCompleted/Year"),
        article.findtext("./MedlineCitation/DateRevised/Year"),
        article.findtext(
            "./MedlineCitation/Article/Journal/JournalIssue/PubDate/MedlineDate"
        ),
    ]
    for value in values:
        match = re.search(r"\b(?:19|20)\d{2}\b", value or "")
        if match:
            return int(match.group(0))
    return None


def parse_pubmed_articles(data: bytes) -> list[dict[str, object]]:
    root = ET.fromstring(data)
    parsed = []
    for item in root.findall(".//PubmedArticle"):
        citation = item.find("MedlineCitation")
        article = citation.find("Article") if citation is not None else None
        if citation is None or article is None:
            continue
        identifiers = {
            (node.attrib.get("IdType") or "").casefold(): normalize_space(node.text)
            for node in item.findall("./PubmedData/ArticleIdList/ArticleId")
        }
        abstract_parts = []
        for node in article.findall("./Abstract/AbstractText"):
            text = element_text(node)
            if not text:
                continue
            label = normalize_space(node.attrib.get("Label"))
            abstract_parts.append(f"{label}: {text}" if label else text)
        pmid = normalize_space(citation.findtext("PMID"))
        parsed.append(
            {
                "pmid": pmid,
                "pmcid": identifiers.get("pmc"),
                "doi": identifiers.get("doi"),
                "title": element_text(article.find("ArticleTitle")),
                "abstract": " ".join(abstract_parts),
                "year": _pubmed_year(item),
                "journal": element_text(article.find("./Journal/Title")),
                "publication_types": [
                    element_text(node)
                    for node in article.findall("./PublicationTypeList/PublicationType")
                ],
                "is_open_access": False,
                "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "provenance": {"pubmed_status": item.findtext("./PubmedData/PublicationStatus")},
            }
        )
    return parsed


def discover_europe_pmc(
    db: sqlite3.Connection, client: ApiClient, max_papers: int
) -> None:
    cursor_mark = "*"
    page = 0
    while db.execute("SELECT COUNT(*) FROM papers").fetchone()[0] < max_papers:
        page += 1
        data, request = client.get(
            "Europe PMC",
            f"{EUROPE_PMC_API}/search",
            {
                "query": EUROPE_PMC_QUERY,
                "format": "json",
                "resultType": "core",
                "pageSize": 1000,
                "cursorMark": cursor_mark,
            },
            "search",
            EUROPE_PMC_QUERY,
            cursor_mark,
        )
        if not data:
            raise RuntimeError(f"Europe PMC request failed: {request['error']}")
        payload = json.loads(data)
        records = payload.get("resultList", {}).get("result", [])
        total = int(payload.get("hitCount") or 0)
        client.annotate(request["request_key"], total, len(records))
        for raw in records:
            allow = db.execute("SELECT COUNT(*) FROM papers").fetchone()[0] < max_papers
            upsert_paper(db, epmc_record(raw), "Europe PMC", allow_insert=allow)
        db.commit()
        print(
            f"Europe PMC page {page}: {db.execute('SELECT COUNT(*) FROM papers').fetchone()[0]}"
            f"/{max_papers} unique papers",
            flush=True,
        )
        next_cursor = payload.get("nextCursorMark")
        if not records or not next_cursor or next_cursor == cursor_mark:
            break
        cursor_mark = next_cursor


def discover_pubmed(
    db: sqlite3.Connection, client: ApiClient, max_papers: int
) -> None:
    def search_ids(
        retmax: int,
        page_token: str,
        mindate: str | None = None,
        maxdate: str | None = None,
    ) -> tuple[int, list[str]]:
        params: dict[str, object] = {
            "db": "pubmed",
            "term": PUBMED_QUERY,
            "retmode": "json",
            "retmax": retmax,
            "sort": "relevance",
        }
        if mindate and maxdate:
            params.update(datetype="pdat", mindate=mindate, maxdate=maxdate)
        data, request = client.get(
            "PubMed",
            f"{PUBMED_API}/esearch.fcgi",
            params,
            "search",
            PUBMED_QUERY,
            page_token,
        )
        if not data:
            raise RuntimeError(f"PubMed search failed: {request['error']}")
        payload = json.loads(data)["esearchresult"]
        result_count = int(payload.get("count") or 0)
        result_ids = list(payload.get("idlist") or [])
        client.annotate(request["request_key"], result_count, len(result_ids))
        return result_count, result_ids

    total, initial_ids = search_ids(
        min(max_papers, PUBMED_MAX_IDS_PER_SEARCH), "all"
    )
    if total <= PUBMED_MAX_IDS_PER_SEARCH:
        identifiers = initial_ids
    else:
        # PubMed ESearch exposes at most 9,999 IDs for one PubMed query.
        # Date shards retrieve the complete result set without bypassing that
        # documented limit. The broad pre-2000 shard is well below the cap for
        # this registered query; every later year is requested independently.
        current_year = dt.datetime.now(dt.timezone.utc).year
        shards = [("1700/01/01", "1999/12/31", "1700-1999")]
        shards.extend(
            (f"{year}/01/01", f"{year}/12/31", str(year))
            for year in range(2000, current_year + 2)
        )
        identifiers = []
        seen_ids: set[str] = set()
        for mindate, maxdate, label in shards:
            shard_count, shard_ids = search_ids(
                PUBMED_MAX_IDS_PER_SEARCH,
                f"publication_date:{label}",
                mindate,
                maxdate,
            )
            if shard_count > PUBMED_MAX_IDS_PER_SEARCH:
                raise RuntimeError(
                    f"PubMed date shard {label} has {shard_count} results; "
                    "use a finer date subdivision."
                )
            for identifier in shard_ids:
                if identifier not in seen_ids:
                    seen_ids.add(identifier)
                    identifiers.append(identifier)
            print(
                f"PubMed search shard {label}: {shard_count}; "
                f"{len(identifiers)}/{total} unique IDs",
                flush=True,
            )
        if len(identifiers) != total:
            print(
                f"PubMed search warning: sharded retrieval returned "
                f"{len(identifiers)} of {total} reported IDs",
                flush=True,
            )
    for offset in range(0, len(identifiers), 200):
        batch = identifiers[offset : offset + 200]
        xml, fetch_request = client.get(
            "PubMed",
            f"{PUBMED_API}/efetch.fcgi",
            {
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml",
                "rettype": "abstract",
            },
            "metadata",
            PUBMED_QUERY,
            str(offset),
        )
        if not xml:
            raise RuntimeError(f"PubMed metadata request failed: {fetch_request['error']}")
        records = parse_pubmed_articles(xml)
        client.annotate(fetch_request["request_key"], len(batch), len(records))
        for record in records:
            allow = db.execute("SELECT COUNT(*) FROM papers").fetchone()[0] < max_papers
            upsert_paper(db, record, "PubMed", allow_insert=allow)
        db.commit()
        if offset % 1000 == 0:
            print(f"PubMed metadata: {offset + len(batch)}/{len(identifiers)}", flush=True)


def _insert_mention(
    db: sqlite3.Connection,
    paper_id: int,
    mention: Mention,
    section_name: str,
    locator: str,
    source_status: str,
) -> None:
    db.execute(
        """INSERT OR IGNORE INTO disease_mentions(
        paper_id,orpha_id,mention_text,normalized_term,mapping_status,
        candidate_orpha_ids_json,section_name,source_locator,source_status,
        char_start,char_end
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
        (
            paper_id,
            mention.orpha_id,
            mention.mention_text,
            mention.normalized_term,
            mention.mapping_status,
            json.dumps(mention.candidate_ids),
            section_name,
            locator,
            source_status,
            mention.start,
            mention.end,
        ),
    )


def scan_metadata_mentions(db: sqlite3.Connection, mapper: DiseaseMapper) -> None:
    """Cheap first pass used to decide which lawful OA full texts to request."""
    db.execute("DELETE FROM disease_mentions")
    for paper in db.execute("SELECT paper_id,title,abstract FROM papers"):
        units = [("Title", "title", paper["title"] or "")]
        if paper["abstract"]:
            units.append(("Abstract", "abstract", paper["abstract"]))
        for section, locator, text in units:
            for mention in mapper.detect(text):
                _insert_mention(
                    db, paper["paper_id"], mention, section, locator, "abstract_only"
                )
    db.commit()


def _license_text(root: ET.Element) -> str | None:
    values = []
    for node in root.findall("./front/article-meta/permissions/license"):
        values.append(element_text(node))
        values.extend(str(value) for value in node.attrib.values())
        for child in node.iter():
            values.extend(str(value) for value in child.attrib.values())
    return normalize_space(" ".join(values)) or None


def parse_full_text_sections(data: bytes) -> tuple[list[tuple[str, str, str]], str | None]:
    root = ET.fromstring(data)
    body = root.find("body")
    if body is None:
        return [], _license_text(root)
    parents = {child: parent for parent in body.iter() for child in parent}
    sections = []
    for index, paragraph in enumerate(body.iter("p"), 1):
        text = element_text(paragraph)
        if not text:
            continue
        section = "Body"
        ancestor = parents.get(paragraph)
        while ancestor is not None:
            if ancestor.tag == "sec":
                title = ancestor.find("title")
                if title is not None and element_text(title):
                    section = element_text(title)
                    break
            ancestor = parents.get(ancestor)
        sections.append((section, f"/article/body/descendant::p[{index}]", text))
    return sections, _license_text(root)


def fetch_candidate_full_texts(
    db: sqlite3.Connection,
    client: ApiClient,
    max_full_text: int = 0,
) -> None:
    candidates = db.execute(
        """SELECT p.* FROM papers p
        WHERE p.is_open_access=1 AND p.pmcid IS NOT NULL
        AND EXISTS(
          SELECT 1 FROM disease_mentions m
          WHERE m.paper_id=p.paper_id AND m.mapping_status='exact'
        )
        ORDER BY p.paper_id"""
    ).fetchall()
    attempted = 0
    for paper in candidates:
        if max_full_text and attempted >= max_full_text:
            break
        if paper["full_text_raw_path"]:
            raw = client.data_dir / paper["full_text_raw_path"]
            if raw.exists():
                continue
        attempted += 1
        data, request = client.get(
            "Europe PMC",
            f"{EUROPE_PMC_API}/{paper['pmcid']}/fullTextXML",
            None,
            "full_text",
            paper["pmcid"],
            paper["pmcid"],
        )
        if not data:
            continue
        try:
            sections, license_text = parse_full_text_sections(data)
        except ET.ParseError:
            continue
        if not sections:
            continue
        db.execute("DELETE FROM fulltext_sections WHERE paper_id=?", (paper["paper_id"],))
        db.executemany(
            """INSERT INTO fulltext_sections(
            paper_id,section_name,source_locator,text
            ) VALUES(?,?,?,?)""",
            [
                (paper["paper_id"], section, locator, text)
                for section, locator, text in sections
            ],
        )
        db.execute(
            """UPDATE papers SET access_status='open_full_text',full_text_raw_path=?,
            license=?,updated_at=? WHERE paper_id=?""",
            (request["raw_path"], license_text, utc_now(), paper["paper_id"]),
        )
        db.commit()
        if attempted % 25 == 0:
            print(f"Open full texts requested: {attempted}/{len(candidates)}", flush=True)


def sentence_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for match in re.finditer(r"[.!?](?=\s+[A-Z0-9]|\s*$)", text):
        end = match.end()
        if normalize_space(text[start:end]):
            spans.append((start, end))
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
    if start < len(text) and normalize_space(text[start:]):
        spans.append((start, len(text)))
    return spans or ([(0, len(text))] if normalize_space(text) else [])


def passage_windows(text: str, width: int = 3) -> Iterable[tuple[str, int, int, str]]:
    spans = sentence_spans(text)
    seen = set()
    for index in range(len(spans)):
        start = spans[index][0]
        end_index = min(len(spans), index + width) - 1
        end = spans[end_index][1]
        passage = normalize_space(text[start:end])
        key = sha256(passage)
        if passage and key not in seen:
            seen.add(key)
            yield passage, start, end, f"sentences:{index + 1}-{end_index + 1}"


def score_passage(text: str) -> tuple[int, str, int]:
    """Return ordinal score, transparent rule name, and rule specificity."""
    for score, rule, pattern, specificity in SCORE_PATTERNS:
        if pattern.search(text):
            return score, rule, specificity
    return 1, "comparative_context_without_strength_anchor", 1


def validate_score(score: int | None, confidence: int) -> None:
    if score is not None and score not in range(5):
        raise ValueError(f"Invalid ordinal score: {score}")
    if confidence not in (1, 2, 3):
        raise ValueError(f"Invalid extraction confidence: {confidence}")


@dataclass(frozen=True)
class Evidence:
    score: int
    rule: str
    specificity: int
    passage: str
    section: str
    locator: str
    access: str


def _paper_units(db: sqlite3.Connection, paper: sqlite3.Row) -> list[dict[str, str]]:
    units = [
        {
            "section": "Title",
            "locator": "title",
            "text": paper["title"] or "",
            "access": "abstract_only",
        }
    ]
    if paper["abstract"]:
        units.append(
            {
                "section": "Abstract",
                "locator": "abstract",
                "text": paper["abstract"],
                "access": "abstract_only",
            }
        )
    units.extend(
        {
            "section": row["section_name"],
            "locator": row["source_locator"],
            "text": row["text"],
            "access": "full_text",
        }
        for row in db.execute(
            """SELECT section_name,source_locator,text FROM fulltext_sections
            WHERE paper_id=? ORDER BY section_id""",
            (paper["paper_id"],),
        )
    )
    return units


def _clip_evidence(text: str, limit: int = 1200) -> str:
    text = normalize_space(text)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def evaluate_papers(db: sqlite3.Connection, mapper: DiseaseMapper) -> None:
    """Rebuild mention, pair, score, rejection, and review tables idempotently."""
    db.execute("DELETE FROM review_queue")
    db.execute("DELETE FROM dimension_scores")
    db.execute("DELETE FROM paper_pairs")
    db.execute("DELETE FROM disease_mentions")
    db.execute("DELETE FROM rejected_records")
    names = {
        row["orpha_id"]: row["preferred_name"]
        for row in db.execute("SELECT orpha_id,preferred_name FROM orpha_entities")
    }
    papers = db.execute("SELECT * FROM papers ORDER BY paper_id").fetchall()
    for number, paper in enumerate(papers, 1):
        pair_evidence: dict[
            tuple[str, str], dict[str, list[Evidence]]
        ] = collections.defaultdict(lambda: collections.defaultdict(list))
        exact_ids: set[str] = set()
        ambiguous_terms: dict[str, tuple[str, ...]] = {}
        for unit in _paper_units(db, paper):
            unit_mentions = mapper.detect(unit["text"])
            for mention in unit_mentions:
                _insert_mention(
                    db,
                    paper["paper_id"],
                    mention,
                    unit["section"],
                    unit["locator"],
                    unit["access"],
                )
                if mention.orpha_id:
                    exact_ids.add(mention.orpha_id)
                else:
                    ambiguous_terms[mention.normalized_term] = mention.candidate_ids
            for passage, start, _end, suffix in passage_windows(unit["text"]):
                if not RELATION_PATTERN.search(passage):
                    continue
                mentions = mapper.detect(passage)
                ids = sorted({m.orpha_id for m in mentions if m.orpha_id})
                if len(ids) < 2 or len(ids) > 20:
                    continue
                relevant = [
                    dimension
                    for dimension, pattern in DIMENSION_PATTERNS.items()
                    if pattern.search(passage)
                ]
                if not relevant:
                    continue
                score, rule, specificity = score_passage(passage)
                locator = f"{unit['locator']}#{suffix};offset={start}"
                for left_index, left in enumerate(ids):
                    for right in ids[left_index + 1 :]:
                        for dimension in relevant:
                            pair_evidence[(left, right)][dimension].append(
                                Evidence(
                                    score,
                                    rule,
                                    specificity,
                                    _clip_evidence(passage),
                                    unit["section"],
                                    locator,
                                    unit["access"],
                                )
                            )
        if ambiguous_terms:
            key = f"mapping:{paper['paper_id']}"
            db.execute(
                """INSERT INTO review_queue(
                queue_key,paper_id,reason,priority,details_json,created_at
                ) VALUES(?,?,?,3,?,?)""",
                (
                    key,
                    paper["paper_id"],
                    "ambiguous_orpha_mapping",
                    json.dumps(
                        {term: list(ids) for term, ids in ambiguous_terms.items()}
                    ),
                    utc_now(),
                ),
            )
        if not pair_evidence:
            if not exact_ids:
                reason = (
                    "ambiguous_mapping_only"
                    if ambiguous_terms
                    else "no_unambiguous_disease_mentions"
                )
            elif len(exact_ids) < 2:
                reason = "fewer_than_two_mapped_diseases"
            else:
                reason = "no_comparative_evidence"
            db.execute(
                """INSERT INTO rejected_records(paper_id,reason,details_json,created_at)
                VALUES(?,?,?,?)""",
                (
                    paper["paper_id"],
                    reason,
                    json.dumps(
                        {
                            "mapped_diseases": sorted(exact_ids),
                            "ambiguous_terms": sorted(ambiguous_terms),
                        }
                    ),
                    utc_now(),
                ),
            )
        for (left, right), by_dimension in sorted(pair_evidence.items()):
            all_evidence = [item for values in by_dimension.values() for item in values]
            best_access = (
                "full_text"
                if any(item.access == "full_text" for item in all_evidence)
                else "abstract_only"
            )
            cursor = db.execute(
                """INSERT INTO paper_pairs(
                paper_id,orpha_id_a,orpha_id_b,pair_key,best_access_status,created_at
                ) VALUES(?,?,?,?,?,?)""",
                (
                    paper["paper_id"],
                    left,
                    right,
                    f"{left}|{right}",
                    best_access,
                    utc_now(),
                ),
            )
            paper_pair_id = int(cursor.lastrowid)
            for dimension in DIMENSIONS:
                candidates = by_dimension.get(dimension, [])
                if candidates:
                    chosen = max(
                        candidates,
                        key=lambda item: (
                            item.specificity,
                            item.access == "full_text",
                            len(item.passage),
                        ),
                    )
                    conflicting = len({item.score for item in candidates}) > 1
                    confidence = (
                        1
                        if chosen.access == "abstract_only" or conflicting
                        else 2
                    )
                    validate_score(chosen.score, confidence)
                    values = (
                        chosen.score,
                        str(chosen.score),
                        chosen.passage,
                        chosen.section,
                        chosen.locator,
                        chosen.access,
                        confidence,
                        chosen.rule
                        + (";conflicting_automated_passages" if conflicting else ""),
                    )
                else:
                    validate_score(None, 1)
                    values = (
                        None,
                        "NA",
                        "No qualifying comparative evidence for this dimension was "
                        "detected in the processed passages.",
                        None,
                        None,
                        "not_assessed",
                        1,
                        "not_assessed",
                    )
                score_cursor = db.execute(
                    """INSERT INTO dimension_scores(
                    paper_pair_id,dimension,ordinal_score,score_label,
                    supporting_passage_or_paraphrase,source_section,source_locator,
                    evidence_access_status,extraction_confidence,scoring_rule,created_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                    (paper_pair_id, dimension, *values, utc_now()),
                )
                score_id = int(score_cursor.lastrowid)
                queue_reason = (
                    "automated_score_unverified"
                    if values[0] is not None
                    else "automated_na_unverified"
                )
                db.execute(
                    """INSERT INTO review_queue(
                    queue_key,paper_id,paper_pair_id,score_id,reason,priority,
                    details_json,created_at
                    ) VALUES(?,?,?,?,?,?,?,?)""",
                    (
                        f"score:{score_id}",
                        paper["paper_id"],
                        paper_pair_id,
                        score_id,
                        queue_reason,
                        3 if "conflicting" in values[7] else 2,
                        json.dumps(
                            {
                                "dimension": dimension,
                                "disease_a": names.get(left),
                                "disease_b": names.get(right),
                            }
                        ),
                        utc_now(),
                    ),
                )
        if number % 250 == 0:
            db.commit()
            print(f"Evaluated papers: {number}/{len(papers)}", flush=True)
    db.commit()


def validate_database(db: sqlite3.Connection, data_dir: Path) -> list[str]:
    errors = []
    if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        errors.append("sqlite_integrity")
    if db.execute("PRAGMA foreign_key_check").fetchall():
        errors.append("foreign_key_check")
    for field in ("pmid", "pmcid", "doi", "normalized_title"):
        duplicate = db.execute(
            f"""SELECT {field} FROM papers WHERE {field} IS NOT NULL AND {field}!=''
            GROUP BY {field} HAVING COUNT(*)>1 LIMIT 1"""
        ).fetchone()
        if duplicate:
            errors.append(f"duplicate_{field}:{duplicate[0]}")
    invalid_rows = db.execute(
        """SELECT paper_pair_id,COUNT(*) FROM dimension_scores
        GROUP BY paper_pair_id HAVING COUNT(*)!=?""",
        (len(DIMENSIONS),),
    ).fetchall()
    errors.extend(f"dimension_count:{row[0]}" for row in invalid_rows)
    invalid_dimensions = db.execute(
        f"""SELECT score_id FROM dimension_scores
        WHERE dimension NOT IN ({','.join('?' * len(DIMENSIONS))})""",
        DIMENSIONS,
    ).fetchall()
    errors.extend(f"invalid_dimension:{row[0]}" for row in invalid_dimensions)
    missing_evidence = db.execute(
        """SELECT score_id FROM dimension_scores WHERE ordinal_score IS NOT NULL
        AND (source_locator IS NULL OR supporting_passage_or_paraphrase IS NULL)"""
    ).fetchall()
    errors.extend(f"missing_evidence:{row[0]}" for row in missing_evidence)
    abstract_confidence = db.execute(
        """SELECT score_id FROM dimension_scores
        WHERE evidence_access_status='abstract_only' AND extraction_confidence!=1"""
    ).fetchall()
    errors.extend(f"abstract_not_low_confidence:{row[0]}" for row in abstract_confidence)
    automated_verified = db.execute(
        """SELECT score_id FROM dimension_scores
        WHERE extraction_method='automated' AND verification_status!='unverified'"""
    ).fetchall()
    errors.extend(f"automated_marked_verified:{row[0]}" for row in automated_verified)
    ambiguous_ids = db.execute(
        """SELECT mention_id FROM disease_mentions
        WHERE mapping_status='ambiguous' AND orpha_id IS NOT NULL"""
    ).fetchall()
    errors.extend(f"ambiguous_id_guessed:{row[0]}" for row in ambiguous_ids)
    for row in db.execute(
        """SELECT request_key,raw_path,response_sha256 FROM search_log
        WHERE status='complete'"""
    ):
        path = data_dir / row["raw_path"]
        if not path.exists() or sha256(path.read_bytes()) != row["response_sha256"]:
            errors.append(f"raw_checksum:{row['request_key']}")
    return errors


def build_report(
    db: sqlite3.Connection, data_dir: Path, requested_limit: int
) -> dict[str, object]:
    papers = db.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    paper_pairs = db.execute("SELECT COUNT(*) FROM paper_pairs").fetchone()[0]
    valid_papers = db.execute(
        "SELECT COUNT(DISTINCT paper_id) FROM paper_pairs"
    ).fetchone()[0]
    unique_pairs = db.execute(
        "SELECT COUNT(DISTINCT pair_key) FROM paper_pairs"
    ).fetchone()[0]
    coverage = {}
    for dimension in DIMENSIONS:
        scored = db.execute(
            """SELECT COUNT(*) FROM dimension_scores
            WHERE dimension=? AND ordinal_score IS NOT NULL""",
            (dimension,),
        ).fetchone()[0]
        coverage[dimension] = {
            "scored": scored,
            "paper_pairs": paper_pairs,
            "percent": round(100 * scored / paper_pairs, 2) if paper_pairs else 0.0,
        }
    score_access = {
        row[0]: row[1]
        for row in db.execute(
            """SELECT evidence_access_status,COUNT(*) FROM dimension_scores
            WHERE ordinal_score IS NOT NULL GROUP BY evidence_access_status"""
        )
    }
    paper_access = {
        row[0]: row[1]
        for row in db.execute(
            "SELECT access_status,COUNT(*) FROM papers GROUP BY access_status"
        )
    }
    rejections = {
        row[0]: row[1]
        for row in db.execute(
            "SELECT reason,COUNT(*) FROM rejected_records GROUP BY reason"
        )
    }
    ambiguous_mentions, ambiguous_papers = db.execute(
        """SELECT COUNT(*),COUNT(DISTINCT paper_id) FROM disease_mentions
        WHERE mapping_status='ambiguous'"""
    ).fetchone()
    unverified = db.execute(
        "SELECT COUNT(*) FROM dimension_scores WHERE verification_status='unverified'"
    ).fetchone()[0]
    unverified_scored = db.execute(
        """SELECT COUNT(*) FROM dimension_scores
        WHERE verification_status='unverified' AND ordinal_score IS NOT NULL"""
    ).fetchone()[0]
    request_status = [
        {
            "source": row[0],
            "request_kind": row[1],
            "status": row[2],
            "count": row[3],
        }
        for row in db.execute(
            """SELECT source,request_kind,status,COUNT(*) FROM search_log
            GROUP BY source,request_kind,status ORDER BY source,request_kind,status"""
        )
    ]
    score_distribution = {
        row[0]: row[1]
        for row in db.execute(
            """SELECT score_label,COUNT(*) FROM dimension_scores
            GROUP BY score_label ORDER BY score_label"""
        )
    }
    errors = validate_database(db, data_dir)
    report = {
        "pipeline_version": PIPELINE_VERSION,
        "generated_at": utc_now(),
        "database": str((data_dir / "papers.sqlite").resolve()),
        "unique_papers_collected": papers,
        "papers_containing_valid_disease_pairs": valid_papers,
        "paper_pair_observations": paper_pairs,
        "unique_mapped_disease_pairs": unique_pairs,
        "score_coverage": coverage,
        "abstract_vs_full_text_coverage": {
            "papers_by_access_status": paper_access,
            "scored_dimensions_by_evidence_access": score_access,
        },
        "ambiguous_mappings": {
            "mentions": ambiguous_mentions,
            "papers": ambiguous_papers,
        },
        "rejected_records": {
            "total": sum(rejections.values()),
            "by_reason": rejections,
        },
        "unverified_rows_requiring_human_review": unverified,
        "unverified_non_na_scores": unverified_scored,
        "review_queue_pending": db.execute(
            "SELECT COUNT(*) FROM review_queue WHERE status='pending'"
        ).fetchone()[0],
        "automated_score_distribution": score_distribution,
        "api_request_status": request_status,
        "requested_paper_limit": requested_limit,
        "found_5000_legitimate_papers": papers >= 5000,
        "found_requested_limit": papers >= requested_limit,
        "validation": {"passed": not errors, "errors": errors},
        "notes": [
            "All papers originate from Europe PMC or PubMed API responses; no synthetic records were added.",
            "All automated scores and automated NA decisions remain unverified.",
            "Abstract-derived scores have extraction confidence 1.",
            "A valid paper pair requires exact unambiguous ORPHA mappings, a comparative cue, and dimension evidence in a short passage.",
        ],
    }
    report_path = data_dir / "acquisition_report.json"
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(report_path)
    return report


def run_pipeline(
    db_path: Path = DEFAULT_DB,
    max_papers: int = 5000,
    max_full_text: int = 0,
) -> dict[str, object]:
    if max_papers < 1 or max_papers > 100000:
        raise ValueError("--max-papers must be between 1 and 100000")
    data_dir = db_path.parent
    db = connect(db_path)
    try:
        import_orphanet(db)
        set_state(db, "requested_paper_limit", max_papers)
        set_state(db, "pipeline_version", PIPELINE_VERSION)
        db.commit()
        client = ApiClient(db, data_dir)
        discover_europe_pmc(db, client, max_papers)
        # PubMed is an independent authoritative search and enriches/deduplicates
        # records even when Europe PMC has already reached the collection cap.
        discover_pubmed(db, client, max_papers)
        mapper = DiseaseMapper.from_db(db)
        scan_metadata_mentions(db, mapper)
        fetch_candidate_full_texts(db, client, max_full_text=max_full_text)
        evaluate_papers(db, mapper)
        errors = validate_database(db, data_dir)
        if errors:
            raise ValueError(f"Database validation failed: {errors[:20]}")
        report = build_report(db, data_dir, max_papers)
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
        return report
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report", "validate"))
    parser.add_argument("--database", type=Path, default=DEFAULT_DB)
    parser.add_argument("--max-papers", type=int, default=5000)
    parser.add_argument(
        "--max-full-text",
        type=int,
        default=0,
        help="Maximum OA full texts to fetch; 0 means every eligible candidate.",
    )
    args = parser.parse_args()
    db_path = args.database.resolve()
    if args.command == "run":
        run_pipeline(db_path, args.max_papers, args.max_full_text)
        return
    db = connect(db_path)
    try:
        if args.command == "validate":
            errors = validate_database(db, db_path.parent)
            print(json.dumps({"passed": not errors, "errors": errors}, indent=2))
            raise SystemExit(1 if errors else 0)
        requested = int(get_state(db, "requested_paper_limit", args.max_papers))
        print(
            json.dumps(
                build_report(db, db_path.parent, requested),
                indent=2,
                ensure_ascii=False,
            )
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
