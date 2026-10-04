"""Blinded expert-review sheet: are the top neighbours clinically or mechanistically related?

Ground truth only covers relations somebody already recorded; the interesting neighbours are the ones nobody has.
This samples query diseases, takes each model's top 5 neighbours (v2 and v3 static), merges and shuffles them so the
reviewer cannot tell which model proposed which, and writes a sheet to rate 0 (unrelated) to 3 (clearly related).
`key.csv` holds the model, rank and whether the pair is already a paper-stated or curated relation.

    python global_eval/review_sheet.py [--queries 60] [--top 5]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core

parser = argparse.ArgumentParser()
parser.add_argument("--queries", type=int, default=60)
parser.add_argument("--top", type=int, default=5)
args = parser.parse_args()

tasks = core.read_json(core.DATA / "tasks.json")
ids, names, category = tasks["catalogue"], tasks["names"], tasks["category"]
col = {d: i for i, d in enumerate(ids)}
queries = sorted({p[k] for p in tasks["paper_pairs"] for k in ("a", "b")})
row = {q: i for i, q in enumerate(queries)}
known = {}
for p in tasks["paper_pairs"]:
    known.setdefault(p["a"], set()).add(p["b"]); known.setdefault(p["b"], set()).add(p["a"])
folder = core.DATA / "scores" / "paper_pairs"
scores = {m: np.load(folder / f"{m}.npy") for m in ("v2_drugfree", "v3_static")}

rng = np.random.default_rng(0)
# Stratify by category so one disease area does not dominate.
by_cat = {}
for q in queries:
    by_cat.setdefault(category.get(q, "?"), []).append(q)
chosen = []
cats = sorted(by_cat)
while len(chosen) < min(args.queries, len(queries)):
    for c in cats:
        pool = [q for q in by_cat[c] if q not in chosen]
        if pool and len(chosen) < args.queries:
            chosen.append(str(rng.choice(pool)))

sheet, key = [], []
for q in chosen:
    proposals = {}
    for m, M in scores.items():
        s = M[row[q]].astype(np.float64).copy(); s[col[q]] = -np.inf
        for rank, j in enumerate(np.argsort(-s)[: args.top], 1):
            proposals.setdefault(ids[j], {})[m] = rank
    for cand in rng.permutation(sorted(proposals)):
        item = f"{q}|{cand}"
        sheet.append({"item": item, "disease": names[q], "disease_orpha": q, "disease_category": category[q],
                      "candidate": names[cand], "candidate_orpha": cand, "candidate_category": category[cand],
                      "rating_0_to_3": "", "notes": ""})
        curated = [k for k, e in tasks["relations"].items() if cand in e["queries"].get(q, [])]
        key.append({"item": item, "v2_rank": proposals[cand].get("v2_drugfree", ""), "v3_static_rank": proposals[cand].get("v3_static", ""),
                    "paper_stated": cand in known.get(q, ()), "curated_in_sample": ";".join(curated)})

out = core.RESULTS / "expert_review"; out.mkdir(parents=True, exist_ok=True)
for name, rows in (("review_sheet.csv", sheet), ("key.csv", key)):
    with (out / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
both = sum(1 for k in key if k["v2_rank"] != "" and k["v3_static_rank"] != "")
print(f"{len(chosen)} diseases, {len(sheet)} candidate pairs ({both} proposed by both models) -> {out}")
