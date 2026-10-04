"""Write the per-disease annotations and the literature pairs the explorer colours and checks against.

Run after ``build_graph_data.py`` (it reuses the node order of ``graph.json``).  Needs the v2 Python
environment because the annotations come from v2's cached knowledge bundle.  Writes to
``frontend/public/data/``:

* ``annotations.json``: for every gene, HPO term (with its ancestors, so "Seizure" includes its subtypes),
  age-of-onset class and inheritance mode, the diseases carrying it (indices into ``graph.json`` nodes).
  Loaded lazily, the first time a colour mode needs it.
* ``literature.json``: the curated paper-pair dataset (``literature_review/additional_runs``), grouped by paper,
  with each stated similarity dimension mapped onto v2's modalities.  The UI checks every claim against the graph.

Usage (from the repository root):
    python3 frontend/scripts/build_annotations.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "frontend" / "public" / "data"
LITERATURE_CSV = PROJECT_ROOT / "literature_review" / "additional_runs" / "literature_disease_pairs_pairfirst.csv"

# Dimension named by the paper -> v2 modalities that would carry it.  Empty: v2 has no modality for it, so only the
# existence of the edge can be checked.
DIMENSION_MODALITIES = {
    "phenotype": ["phenotype"],
    "genes": ["gene", "ot_gene"],
    "pathways": ["pathway"],
    "treatment": ["drug", "drug_target"],
    "epidemiology": ["prevalence", "onset", "inheritance"],
    "other (diagnostic)": [],
    "other (comorbidity)": [],
}

csv.field_size_limit(sys.maxsize)


def annotations(node_ids: list[str]) -> dict:
    sys.path.insert(0, str(PROJECT_ROOT / "v2"))
    import data_sources

    bundle = data_sources.load_bundle()
    knowledge = bundle.knowledge
    index = {i: n for n, i in enumerate(node_ids)}

    genes: dict[str, list[int]] = defaultdict(list)
    phenotypes: dict[str, dict[int, float]] = defaultdict(dict)
    onset: dict[str, list[int]] = defaultdict(list)
    inheritance: dict[str, list[int]] = defaultdict(list)
    skipped = 0
    for disease_id, record in bundle.records.items():
        n = index.get(disease_id)
        if n is None:
            skipped += 1
            continue
        for gene in record.get("genes", {}):
            genes[gene].append(n)
        for term, weight in (record.get("phenotypes") or {}).items():
            for ancestor in knowledge.hpo_ancestors(term):
                if knowledge.hpo_labels.get(ancestor) is None:
                    continue
                current = phenotypes[ancestor].get(n, 0.0)
                phenotypes[ancestor][n] = max(current, float(weight))
        for value in record.get("onset", []):
            if value != "No data available":
                onset[value].append(n)
        for value in record.get("inheritance", []):
            if value not in ("No data available", "Not yet documented", "Undetermined", "Unknown"):
                inheritance[value].append(n)
    print(f"{len(genes):,} genes, {len(phenotypes):,} HPO terms, {len(onset)} onset classes, "
          f"{len(inheritance)} inheritance modes ({skipped} records not in the graph)")
    return {
        "nodes": node_ids,
        "genes": {g: sorted(v) for g, v in sorted(genes.items())},
        "phenotypes": {
            t: {
                "l": knowledge.hpo_labels[t],
                "n": sorted(w),
                # Share of patients, in tenths (0-10); an ancestor takes its most frequent descendant.
                "f": [round(w[i] * 10) for i in sorted(w)],
            }
            for t, w in sorted(phenotypes.items())
        },
        "onset": {k: sorted(v) for k, v in onset.items()},
        "inheritance": {k: sorted(v) for k, v in inheritance.items()},
    }


def edge_shard(edge_id: str, shards: int) -> int:
    """djb2 over the ID, as in build_graph_data.py and src/data/source.ts."""

    h = 5381
    for ch in edge_id:
        h = ((h * 33) + ord(ch)) & 0xFFFFFFFF
    return h % shards


def literature(graph: dict) -> dict:
    """Papers with, for every claim, the graph's edge for that pair (if any) and its compact explanation.

    Baking the explanation in lets the UI judge all claims without fetching the 64 detail shards.
    """

    ids = [n["id"] for n in graph["nodes"]]
    node_ids = set(ids)
    edge_ids = {}
    for s, t, *_ in graph["edges"]:
        edge_ids[(ids[s], ids[t])] = f"{ids[s]}|{ids[t]}"
        edge_ids[(ids[t], ids[s])] = f"{ids[s]}|{ids[t]}"
    shards: dict[int, dict] = {}

    def detail(edge_id: str):
        shard = edge_shard(edge_id, graph["shards"])
        if shard not in shards:
            shards[shard] = json.loads((OUTPUT_DIR / "details" / f"{shard:02d}.json").read_text(encoding="utf-8"))
        return shards[shard].get(edge_id)

    if not LITERATURE_CSV.is_file():
        print(f"{LITERATURE_CSV} not found; writing an empty literature set")
        return {"papers": [], "dimensions": DIMENSION_MODALITIES}
    papers: dict[str, dict] = {}
    dropped = 0
    for row in csv.DictReader(LITERATURE_CSV.open(encoding="utf-8")):
        a, b = row["disease_a_orpha_id"], row["disease_b_orpha_id"]
        if a not in node_ids or b not in node_ids or a == b:
            dropped += 1
            continue
        key = row["pmid"] or row["link"] or row["paper_title"]
        paper = papers.setdefault(
            key,
            {
                "id": f"pmid:{row['pmid']}" if row["pmid"] else key,
                "title": row["paper_title"],
                "year": int(row["year"]) if row["year"].isdigit() else None,
                "link": row["link"],
                "studyType": row["study_type"],
                "claims": [],
            },
        )
        dimensions = [d.strip() for d in row["similarity_dimension"].split(";") if d.strip()]
        paper["claims"].append(
            {
                "a": a,
                "b": b,
                "dimensions": dimensions,
                "relationship": row["relationship"],
                "stated": float(row["similarity_score"]) if row["similarity_score"] else None,
                "finding": row["finding"],
                "lowEvidence": row["low_evidence"] == "yes",
            }
        )
        edge_id = edge_ids.get((a, b))
        if edge_id:
            paper["claims"][-1]["edge"] = edge_id
            paper["claims"][-1]["detail"] = detail(edge_id)
    ordered = sorted(papers.values(), key=lambda p: (-len(p["claims"]), p["title"]))
    claims = [c for p in ordered for c in p["claims"]]
    print(f"{len(ordered)} papers, {len(claims)} claims ({dropped} rows dropped: disease not in graph); "
          f"{sum('edge' in c for c in claims)} claims have an edge in the graph")
    return {"papers": ordered, "dimensions": DIMENSION_MODALITIES}


def main() -> int:
    graph = json.loads((OUTPUT_DIR / "graph.json").read_text(encoding="utf-8"))
    node_ids = [n["id"] for n in graph["nodes"]]
    (OUTPUT_DIR / "annotations.json").write_text(
        json.dumps(annotations(node_ids), separators=(",", ":"), ensure_ascii=False), encoding="utf-8"
    )
    (OUTPUT_DIR / "literature.json").write_text(
        json.dumps(literature(graph), separators=(",", ":"), ensure_ascii=False), encoding="utf-8"
    )
    for name in ("annotations.json", "literature.json"):
        print(f"{name}: {(OUTPUT_DIR / name).stat().st_size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
