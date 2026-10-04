"""Place a new disease into the v2 rare-disease similarity graph.

A new disease is described in JSON with any subset of these fields (all
optional except ``name``):

    {
      "name": "...", "synonyms": ["..."], "description": "free text",
      "phenotypes": ["HP:0001250", ...] or {"HP:0001250": 0.9, ...},
      "genes": ["CDKL5"], "drugs": ["CHEMBL1234" or "drug name"],
      "inheritance": ["X-linked dominant"], "onset": ["Infancy"],
      "prevalence": 1e-6, "ontology_parents": ["ORPHA:102369"]
    }

Examples:
  python v2/place_disease.py --json v2/examples/new_disease_example.json
  python v2/place_disease.py --json v2/examples/new_disease_example.json --add
  python v2/place_disease.py --orpha ORPHA:558 --hide ontology,name

``--orpha`` re-places an existing catalogue disease: it is removed from the
candidates and ``--hide`` can withhold modalities (e.g. its Orphanet
classification) to mimic a newly described, not yet classified disease. The
production model was fitted on the whole catalogue, including that disease;
held-out performance on unseen diseases is measured by ``run_evaluation.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from common import GRAPH_DIR, configure_stdout, normalize_orpha_id
from data_sources import load_bundle, record_from_user_input
from modalities import MODALITIES
from production import ProductionModel, load_model

USER_DISEASES = GRAPH_DIR / "user_diseases.jsonl"
RELATION_TITLES = {
    "orphanet_siblings": "Orphanet relation",
    "shared_causal_gene": "shared causal gene",
    "shared_drug": "shared trial drug",
}


def load_user_diseases() -> list[dict[str, Any]]:
    if not USER_DISEASES.is_file():
        return []
    return [json.loads(line) for line in USER_DISEASES.read_text(encoding="utf-8").splitlines() if line.strip()]


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:60] or "new-disease"


def place(
    model: ProductionModel,
    record: dict[str, Any],
    query_id: str,
    top: int = 10,
    hide: Sequence[str] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rank the catalogue for one disease; returns the neighbours and the scoring details."""

    scored = model.score_record(record, hide)
    S, A, fusion = scored["S"], scored["A"], scored["fusion"]
    scores = scored["logits"].copy()
    if query_id in model.engine.index:
        scores[model.engine.index[query_id]] = -np.inf
    order = np.argsort(-scores)[:top]
    percentiles = model.percentile(scores[order], fusion)
    results = []
    for rank, (j, percentile) in enumerate(zip(order, percentiles), start=1):
        neighbor = model.ids[j]
        results.append(
            {
                "rank": rank,
                "id": neighbor,
                "name": model.names[j],
                "category": model.top_categories[j],
                "score": round(float(scores[j]), 4),
                "percentile": round(float(percentile), 6),
                "explanation": model.explain(scored["encoded"], 0, int(j), S[j], A[j], fusion=fusion),
                "known_relations": model.known_relations(query_id, neighbor) if query_id.startswith("ORPHA:") else {},
                "similarities": {m: round(float(S[j, c]), 4) for c, m in enumerate(model.engine.modalities) if A[j, c]},
                "contributions": {
                    m: round(float(v), 4) for m, v in zip(model.engine.modalities, fusion.similarity_contributions(S[j]))
                },
                "annotation_adjustment": round(float(fusion.availability_adjustment(A[j])), 4),
            }
        )
    return results, scored


def print_results(name: str, results: list[dict[str, Any]], scored: dict[str, Any], hide: Sequence[str]) -> None:
    print(f"\nNearest diseases to: {name}")
    print(f"Modalities used: {', '.join(scored['present']) or 'none'}" + (f" (hidden: {', '.join(hide)})" if hide else ""))
    weights = [
        f"{m} {w:.2f}" for m, w in zip(scored["fusion"].modalities, scored["fusion"].similarity_weights) if w > 0
    ]
    print(f"Fusion weights for these modalities: {', '.join(weights)}")
    for item in results:
        known = "; ".join(f"{RELATION_TITLES[k]}: {v}" for k, v in item["known_relations"].items())
        print(f"\n{item['rank']:>2}. {item['name']} ({item['id']}) -- {item['category']}")
        print(f"    score {item['score']:+.2f}, higher than {100 * item['percentile']:.2f}% of random disease pairs")
        for part in item["explanation"]:
            shared = "; ".join(part["shared"]) or "-"
            print(f"    + {part['modality']:<11} sim {part['similarity']:.2f}  contribution {part['contribution']:+.2f}  [{shared}]")
        if known:
            print(f"    curated: {known}")


def add_to_graph(model: ProductionModel, query_id: str, record: dict[str, Any], results: list[dict[str, Any]], available: list[str]) -> None:
    nodes_path, edges_path = GRAPH_DIR / "nodes.csv", GRAPH_DIR / "edges.csv"
    if not nodes_path.is_file() or not edges_path.is_file():
        raise FileNotFoundError("Graph files not found. Run `python v2/build_graph.py` first.")
    nodes = pd.read_csv(nodes_path, dtype=str, keep_default_na=False)
    if query_id in set(nodes["orpha_id"]):
        raise ValueError(f"{query_id} is already in the graph; choose another 'id' in the JSON.")
    edges = pd.read_csv(edges_path, dtype=str, keep_default_na=False)
    new_edges = []
    for item in results:
        sims = item["similarities"]
        shares = any(sims.get(m, 0) > 0 for m in ("gene", "drug"))
        contributions = {e["modality"]: e["contribution"] for e in item["explanation"]}
        row = {column: "" for column in edges.columns}
        row.update(
            {
                "source": query_id,
                "target": item["id"],
                "source_name": record["name"],
                "target_name": item["name"],
                "score": item["score"],
                "percentile": item["percentile"],
                "rank_source_to_target": item["rank"],
                "mutual": False,
                "support": "plausible" if shares else "novel",
                "top_modalities": "; ".join(f"{m}:{c:.2f}" for m, c in contributions.items()),
                "evidence": " | ".join(
                    f"{e['modality']} (+{e['contribution']:.2f}): {'; '.join(e['shared'])}" for e in item["explanation"]
                ),
                "annotation_adjustment": item["annotation_adjustment"],
                **{f"sim_{m}": v for m, v in sims.items()},
                **{f"contrib_{m}": v for m, v in item["contributions"].items()},
            }
        )
        new_edges.append(row)
    node = {column: "" for column in nodes.columns}
    node.update(
        {
            "orpha_id": query_id,
            "name": record["name"],
            "top_category": "User-added disease",
            "disorder_type": "User-added disease",
            "degree": len(results),
            "n_modalities": len(available),
            "modalities": ";".join(available),
            "source_type": "user",
        }
    )
    nodes = pd.concat([nodes, pd.DataFrame([node])], ignore_index=True)
    edges = pd.concat([edges, pd.DataFrame(new_edges)], ignore_index=True)
    nodes.to_csv(nodes_path, index=False)
    edges.to_csv(edges_path, index=False, quoting=csv.QUOTE_MINIMAL)
    with USER_DISEASES.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"id": query_id, "name": record["name"], "record": record}) + "\n")

    from build_graph import write_graph

    nodes["degree"] = pd.to_numeric(nodes["degree"])
    edges["score"] = pd.to_numeric(edges["score"])
    edges["percentile"] = pd.to_numeric(edges["percentile"])
    edges["mutual"] = edges["mutual"].astype(str).str.lower() == "true"
    write_graph(nodes, edges)
    print(f"\nAdded {query_id} with {len(new_edges)} edges to {GRAPH_DIR}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_stdout()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--json", type=Path, help="JSON file describing the new disease.")
    source.add_argument("--orpha", help="Existing ORPHA ID to re-place as if it were new (excluded from candidates).")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--hide", default="", help=f"Comma-separated modalities to withhold: {', '.join(MODALITIES)}")
    parser.add_argument("--add", action="store_true", help="Insert the new disease and its edges into the graph files.")
    parser.add_argument("--output", type=Path, help="Write the ranked neighbours as JSON.")
    args = parser.parse_args(argv)

    hide = [m.strip() for m in args.hide.split(",") if m.strip()]
    unknown = sorted(set(hide) - set(MODALITIES))
    if unknown:
        parser.error(f"unknown modalities in --hide: {unknown}")

    model = load_model()
    users = load_user_diseases()
    model.extend([u["id"] for u in users], [u["name"] for u in users], [u["record"] for u in users])

    if args.orpha:
        if args.add:
            parser.error("--add is only for new diseases given with --json")
        query_id = normalize_orpha_id(args.orpha)
        bundle = load_bundle()
        if query_id not in bundle.records:
            parser.error(f"{query_id} is not in the v2 cohort")
        record = bundle.records[query_id]
        warnings: list[str] = []
    else:
        document = json.loads(args.json.read_text(encoding="utf-8"))
        record, warnings = record_from_user_input(document, model.knowledge)
        if not record["name"]:
            parser.error("the JSON needs a 'name'")
        query_id = str(document.get("id") or f"USER:{slug(record['name'])}")
    for warning in warnings:
        print(f"warning: {warning}")

    results, scored = place(model, record, query_id, args.top, hide)
    print_results(record["name"], results, scored, hide)
    if args.orpha:
        known_total = sum(1 for evidence in model.relation_evidence.values() for pair in evidence if query_id in pair)
        recovered = sum(1 for item in results if item["known_relations"])
        print(f"\n{recovered} of the top {args.top} have a curated relation to {query_id} "
              f"({known_total} curated relations in total across benchmarks).")
    if args.output:
        args.output.write_text(json.dumps({"query": query_id, "name": record["name"], "neighbors": results}, indent=2), encoding="utf-8")
    if args.add:
        add_to_graph(model, query_id, record, results, scored["present"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
