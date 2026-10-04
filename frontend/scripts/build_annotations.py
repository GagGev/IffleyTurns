"""Write the per-disease annotations and the literature pairs the explorer colours and checks against.

Run after ``build_graph_data.py`` (it reuses the node order of ``graph.json``).  Needs the v2 Python
environment because the annotations come from v2's cached knowledge bundle.  Writes to
``frontend/public/data/``:

* ``annotations.json``: for every gene, HPO term (with its ancestors, so "Seizure" includes its subtypes),
  age-of-onset class and inheritance mode, the diseases carrying it (indices into ``graph.json`` nodes).
  Loaded lazily, the first time a colour mode needs it.
* ``literature.json``: the curated paper-pair dataset (``literature_review/additional_runs``), grouped by paper,
  with each stated similarity dimension mapped onto v2's modalities.  The UI checks every claim against the graph.
  Under ``acquired`` it also lists the papers of the latest acquisition release
  (``.data/literature_acquisition/v1``), each with the disease pairs its full text co-mentions (unreviewed), so
  "Find a paper" can search both.

Usage (from the repository root):
    python3 frontend/scripts/build_annotations.py
"""

from __future__ import annotations

import csv
import html
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "frontend" / "public" / "data"
LITERATURE_CSV = PROJECT_ROOT / "literature_review" / "additional_runs" / "literature_disease_pairs_pairfirst.csv"
ACQUISITION_DIR = PROJECT_ROOT / ".data" / "literature_acquisition" / "v1"
# Longest passage kept for a co-mention; the median full-text paragraph is about 1,100 characters.
PASSAGE_CHARS = 360

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
    acquired, release = acquired_papers(node_ids, edge_ids, detail, {p["id"] for p in ordered})
    return {"papers": ordered, "dimensions": DIMENSION_MODALITIES, "acquired": acquired, "acquiredRelease": release}


def acquired_papers(node_ids: set[str], edge_ids: dict, detail, known: set[str]) -> tuple[list[dict], str | None]:
    """Every paper of the acquisition release that is not already in the curated set.

    The release has no reviewed scores.  Its review queue holds candidate passages in which open full text names two
    catalogued diseases; these become claims with no dimensions, so the UI only checks whether the graph has the edge.
    """

    exports = ACQUISITION_DIR / "exports"
    if not (exports / "screening_queue.csv").is_file():
        print(f"{exports / 'screening_queue.csv'} not found; no acquisition papers")
        return [], None
    manifest = json.loads((ACQUISITION_DIR / "release_manifest.json").read_text(encoding="utf-8"))
    passages: dict[str, dict[tuple[str, str], dict]] = defaultdict(dict)
    for line in (exports / "review_queue.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        a, b = row["pair_id"].split("|")
        if a not in node_ids or b not in node_ids or a == b or (a, b) in passages[row["paper_id"]]:
            continue
        text = " ".join(row["source_passage"].split())
        hints = ", ".join(d.replace("_", " ") for d in json.loads(row["dimensions_json"] or "[]"))
        claim = {
            "a": a,
            "b": b,
            "dimensions": [],
            "relationship": f"co-mentioned, unreviewed{f' (hint: {hints})' if hints else ''}",
            "stated": None,
            "finding": text if len(text) <= PASSAGE_CHARS else f"{text[:PASSAGE_CHARS].rsplit(' ', 1)[0]}…",
            "locator": row["section_name"] or row["source_locator"],
        }
        edge_id = edge_ids.get((a, b))
        if edge_id:
            claim["edge"] = edge_id
            claim["detail"] = detail(edge_id)
        passages[row["paper_id"]][(a, b)] = claim

    out = []
    for row in csv.DictReader((exports / "screening_queue.csv").open(encoding="utf-8")):
        paper_id = f"pmid:{row['pmid']}" if row["pmid"] else row["paper_id"]
        if paper_id in known:
            continue
        link = (
            f"https://europepmc.org/article/MED/{row['pmid']}" if row["pmid"]
            else f"https://europepmc.org/article/PMC/{row['pmcid']}" if row["pmcid"] else ""
        )
        out.append(
            {
                "id": paper_id,
                # Europe PMC titles keep inline markup, escaped: "&lt;i&gt;RAI1&lt;/i&gt;".
                "title": re.sub(r"<[^>]+>", "", html.unescape(html.unescape(row["title"]))).strip(),
                "year": int(row["year"]) if row["year"].isdigit() else None,
                "link": link,
                "access": row["access_class"],
                "claims": list(passages.get(row["paper_id"], {}).values()),
            }
        )
    out.sort(key=lambda p: (-len(p["claims"]), -(p["year"] or 0), p["title"]))
    print(f"{len(out)} acquisition papers not in the curated set; "
          f"{sum(1 for p in out if p['claims'])} with {sum(len(p['claims']) for p in out)} co-mentioned pairs")
    return out, manifest.get("generated_at", "")[:10] or None


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
