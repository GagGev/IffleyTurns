"""Pair-first Europe PMC search (PAPER_SCRAPING_METHOD.md, step 3).

For every pair in pairs_selected.csv, build queries from the Orphanet name and
synonyms of both diseases, run a general query plus one query per thin
dimension, keep the top hits, deduplicate by PMID and drop papers that do not
mention both diseases in the title or abstract.

Output: one JSON line per (pair, paper) in candidate_papers.jsonl, including the
abstract so it can be scored against the rubric.
"""
import argparse
import csv
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
CACHE = HERE.parent.parent / ".data" / "europepmc_cache"

QUERY_TYPES = {
    "general": "",
    "therapeutic": ' AND (treatment OR therapy OR response)',
    "natural_history": ' AND (survival OR "age at onset" OR prevalence OR incidence)',
    "comorbidity": ' AND (comorbid* OR coexist* OR association)',
    "genetic": ' AND (gene OR mutation OR variant)',
    "mechanism": ' AND (pathway OR pathogenesis)',
}
SKIP_TYPES = {
    "comment", "editorial", "letter", "news", "retraction of publication",
    "retracted publication", "published erratum", "erratum", "preprint",
}
TOP_N = 5
MAX_PAPERS_PER_PAIR = 5


def name_variants(names: list[str], limit: int = 4) -> list[str]:
    out, seen = [], set()
    for n in names:
        n = n.strip()
        key = n.casefold()
        if not n or key in seen or len(n) < 8 or len(n) > 70 or '"' in n:
            continue
        # skip acronym-like synonyms (e.g. "TGCT", "FIME", "LGMD1"): they match unrelated papers
        if re.fullmatch(r"[A-Z0-9][A-Z0-9 \-/]{1,14}", n) or (" " not in n and "-" not in n and n.upper() == n):
            continue
        seen.add(key)
        out.append(n)
        if len(out) >= limit:
            break
    return out


def pair_names(p: dict) -> tuple[list[str], list[str]] | None:
    """Query names for each side with names shared by both diseases removed.
    Returns None when one disease's primary name is also a name of the other (same entity)."""
    na = name_variants([p["disease_a"]] + p["synonyms_a"].split("|"))
    nb = name_variants([p["disease_b"]] + p["synonyms_b"].split("|"))
    la, lb = {n.casefold() for n in na}, {n.casefold() for n in nb}
    if p["disease_a"].casefold() in lb or p["disease_b"].casefold() in la:
        return None
    na = [n for n in na if n.casefold() not in lb]
    nb = [n for n in nb if n.casefold() not in la]
    # a name that merely contains the other side's name (e.g. "X syndrome type 1" / "X syndrome") is not distinguishing
    na = [n for n in na if not any(m.casefold() in n.casefold() or n.casefold() in m.casefold() for m in nb)]
    nb = [n for n in nb if not any(m.casefold() in n.casefold() or n.casefold() in m.casefold() for m in na)]
    return (na, nb) if na and nb else None


def clause(names: list[str]) -> str:
    return "(" + " OR ".join(f'TITLE_ABS:"{n}"' for n in names) + ")"


def search(query: str, session: requests.Session) -> list[dict]:
    CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(query.encode()).hexdigest()
    cached = CACHE / f"{key}.json"
    if cached.exists():
        return json.loads(cached.read_text())
    params = {
        "query": query + " AND HAS_ABSTRACT:y AND LANG:eng",
        "resultType": "core",
        "format": "json",
        "pageSize": TOP_N,
        "sort": "CITED desc",
    }
    for attempt in range(4):
        try:
            r = session.get(API, params=params, timeout=60)
            r.raise_for_status()
            res = r.json().get("resultList", {}).get("result", [])
            cached.write_text(json.dumps(res))
            time.sleep(0.15)
            return res
        except (requests.RequestException, ValueError):
            time.sleep(2 * (attempt + 1))
    return []


def mentions(text: str, names: list[str]) -> bool:
    return any(re.search(r"(?<![A-Za-z0-9])" + re.escape(n) + r"(?![A-Za-z0-9])", text, re.I) for n in names)


def related_in_text(title: str, abstract: str, na: list[str], nb: list[str]) -> bool:
    """Both diseases in the title, or one in the title and one in the abstract,
    or both in one abstract sentence (drops papers that mention one disease in passing)."""
    if mentions(title, na) and mentions(title, nb):
        return True
    if (mentions(title, na) and mentions(abstract, nb)) or (mentions(title, nb) and mentions(abstract, na)):
        return True
    for sent in re.split(r"(?<=[.;!?])\s+", re.sub(r"<[^>]+>", " ", abstract)):
        if mentions(sent, na) and mentions(sent, nb):
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(HERE / "pairs_selected.csv"))
    ap.add_argument("--out", default=str(HERE / "candidate_papers.jsonl"))
    args = ap.parse_args()

    session = requests.Session()
    session.headers["User-Agent"] = "rare-disease-similarity-research"
    pairs = list(csv.DictReader(open(args.pairs)))
    n_kept = 0
    with open(args.out, "w") as out:
        for i, p in enumerate(pairs, 1):
            names = pair_names(p)
            if names is None:
                continue
            na, nb = names
            base = f"{clause(na)} AND {clause(nb)}"
            found: dict[str, dict] = {}
            for qtype, suffix in QUERY_TYPES.items():
                for rec in search(base + suffix, session):
                    pmid = rec.get("pmid")
                    if not pmid or pmid in found:
                        continue
                    types = {t.casefold() for t in (rec.get("pubTypeList", {}) or {}).get("pubType", [])}
                    if types & SKIP_TYPES:
                        continue
                    text = f"{rec.get('title', '')} {rec.get('abstractText', '')}"
                    if len(rec.get("abstractText", "")) < 400:
                        continue
                    if not (mentions(text, na) and mentions(text, nb)):
                        continue
                    if not related_in_text(rec.get("title", ""), rec.get("abstractText", ""), na, nb):
                        continue
                    rec["_query_type"] = qtype
                    found[pmid] = rec
            # cap papers per pair; prefer more-cited, so heavily studied pairs do not dominate
            # round-robin over query types so thin dimensions are represented; cited papers first within a type
            by_type: dict[str, list[dict]] = {}
            for rec in sorted(found.values(), key=lambda r: -(r.get("citedByCount") or 0)):
                by_type.setdefault(rec["_query_type"], []).append(rec)
            kept = []
            while len(kept) < MAX_PAPERS_PER_PAIR and any(by_type.values()):
                for qt in QUERY_TYPES:
                    if by_type.get(qt) and len(kept) < MAX_PAPERS_PER_PAIR:
                        kept.append(by_type[qt].pop(0))
            for rec in kept:
                out.write(json.dumps({
                    "pair_id": p["pair_id"], "stratum": p.get("stratum", ""),
                    "disease_a": p["disease_a"], "disease_b": p["disease_b"],
                    "pmid": rec["pmid"], "pmcid": rec.get("pmcid", ""),
                    "title": rec.get("title", ""), "year": rec.get("pubYear", ""),
                    "doi": rec.get("doi", ""),
                    "open_access": rec.get("isOpenAccess", "N"),
                    "in_pmc": rec.get("inPMC", "N"),
                    "pub_types": (rec.get("pubTypeList", {}) or {}).get("pubType", []),
                    "query_type": rec["_query_type"],
                    "abstract": rec.get("abstractText", ""),
                }) + "\n")
                n_kept += 1
            if i % 25 == 0:
                print(f"{i}/{len(pairs)} pairs, {n_kept} candidate papers", file=sys.stderr, flush=True)
    print(f"done: {n_kept} candidate (pair, paper) rows", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
