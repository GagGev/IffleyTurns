"""Paper-stated disease pairs used by the global evaluation (merged from both literature datasets).

Both literature datasets are merged.  A pair counts once; its stated similarity is the mean over the papers that
discuss it, and it is "similar" when any paper calls it that.  Pairs whose ORPHA mapping is ambiguous are dropped.
"""
import csv
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # global_eval/scorers/ -> repo root
SOURCES = [
    ROOT / "literature_review" / "literature_disease_pairs.csv",
    ROOT / "literature_review" / "additional_runs" / "literature_disease_pairs_pairfirst.csv",
]
ID = re.compile(r"^ORPHA:\d+$")

csv.field_size_limit(10**9)


def load_pairs(valid_ids: set[str]) -> list[dict]:
    agg: dict[tuple[str, str], dict] = {}
    for source in SOURCES:
        for row in csv.DictReader(source.open(encoding="utf-8")):
            a, b = row["disease_a_orpha_id"].strip(), row["disease_b_orpha_id"].strip()
            if not (ID.match(a) and ID.match(b)) or a == b or a not in valid_ids or b not in valid_ids:
                continue
            if row["relationship"] == "unrelated" or not row["similarity_score"]:
                continue
            key = tuple(sorted((a, b)))
            entry = agg.setdefault(key, {"a": key[0], "b": key[1], "scores": [], "similar": False, "dims": set(), "papers": set()})
            entry["scores"].append(float(row["similarity_score"]))
            entry["similar"] |= row["relationship"] == "similar"
            entry["dims"].update(d.strip() for d in row["similarity_dimension"].split(";") if d.strip())
            entry["papers"].add(row["pmid"] or row["paper_title"])
    pairs = []
    for entry in agg.values():
        entry["stated"] = sum(entry["scores"]) / len(entry["scores"])
        entry["dims"] = sorted(entry["dims"])
        entry["papers"] = len(entry["papers"])
        del entry["scores"]
        pairs.append(entry)
    return sorted(pairs, key=lambda p: (p["a"], p["b"]))
