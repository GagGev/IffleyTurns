"""Place a new disease into the v3 rare-disease graph.

A new disease is described in JSON with any subset of these fields (all
optional except ``name``):

    {
      "name": "...", "synonyms": ["..."], "description": "free text",
      "phenotypes": ["HP:0001250", ...] or {"HP:0001250": 0.9, ...},
      "genes": ["CDKL5"], "drugs": ["CHEMBL1234" or "drug name"],
      "inheritance": ["X-linked dominant"], "onset": ["Infancy"],
      "prevalence": 1e-6, "ontology_parents": ["ORPHA:102369"],
      "oncology": false
    }

``drugs`` are treated as orphan-designated for the new disease; they feed the
forecast (shared targets, genes targeted by other diseases' drugs) but never
the similarity, which is drug-free.

Examples:
  python v3/place_disease.py --json v3/examples/new_disease_example.json
  python v3/place_disease.py --json v3/examples/new_disease_example.json --add
  python v3/place_disease.py --orpha ORPHA:558 --hide ontology,name
  python v3/place_disease.py --json ... --rank-by forecast

``--orpha`` re-places an existing node as if it were new: it is removed from
the candidates, its designations and regulatory relations are ignored, and
``--hide`` can withhold modalities (e.g. its Orphanet classification) to mimic
a newly described, not yet classified disease.
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

from common import GRAPH_DIR, configure_stdout, normalize_orpha_id, pair_key
from data_sources import record_from_user_input
from modalities import MODALITIES
from production import ProductionModel, load_model

USER_DISEASES = GRAPH_DIR / "user_diseases.jsonl"


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
    rank_by: str = "similarity",
    oncology: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rank every node for one disease; returns the neighbours and the scoring details."""

    world = model.world
    scored = model.score_record(record, hide, oncology)
    block = scored["block"]
    similarity = block["static_logit"][0].copy()
    forecast = block["forecast"][0].copy()
    excluded = np.zeros(world.n, dtype=bool)
    if query_id in world.index:
        q = world.index[query_id]
        excluded[q] = True
        excluded |= world.nested[q].toarray().ravel() > 0
    primary = (similarity if rank_by == "similarity" else forecast).copy()
    primary[excluded] = -np.inf
    order = np.argsort(-primary)[:top]
    S, A = scored["S"][order], scored["A"][order]
    occlusion = model.neural_occlusion(
        np.zeros(len(order), dtype=np.int64), order, S, A, query_z=scored["z"], query_record=record, query_hide=hide
    )
    sim_pct = model.similarity_percentile(similarity[order])
    fc_pct = model.forecast_percentile(forecast[order])
    relations = {pair_key(r.a, r.b): r for r in world.regulatory.relations.itertuples(index=False)}
    results = []
    for rank, j in enumerate(order, start=1):
        neighbor = world.ids[j]
        known = relations.get(pair_key(query_id, neighbor)) if query_id in world.index else None
        results.append(
            {
                "rank": rank,
                "id": neighbor,
                "name": world.name(neighbor),
                "level": world.bundle.meta.at[neighbor, "level"],
                "category": world.top_category[j],
                "similarity": round(float(similarity[j]), 4),
                "similarity_percentile": round(float(sim_pct[rank - 1]), 6),
                "forecast": round(float(forecast[j]), 4),
                "forecast_percentile": round(float(fc_pct[rank - 1]), 6),
                "static_evidence": model.static_evidence(scored["encoded"], 0, int(j), S[rank - 1], A[rank - 1], occlusion[rank - 1]),
                "regulatory_evidence": model.regulatory_evidence(None, int(j), record.get("drugs", {}), record.get("genes", {})),
                "known_regulatory_relation": {"year": int(known.year), "drugs": list(known.drugs)[:4]} if known is not None else None,
                "similarities": {m: round(float(S[rank - 1, c]), 4) for c, m in enumerate(MODALITIES) if A[rank - 1, c]},
            }
        )
    return results, scored


def compare_user_diseases(model: ProductionModel, record: dict[str, Any], users: list[dict[str, Any]], hide: Sequence[str]) -> list[dict[str, Any]]:
    """Static similarity to previously added user diseases (not part of the catalogue)."""

    if not users:
        return []
    encoded_q, available_q = model.encode_record(record, hide)
    z_q = model.static.neural.embeddings(encoded_q, available_q)
    out = []
    for user in users:
        encoded_u, available_u = model.encode_record(user["record"])
        z_u = model.static.neural.embeddings(encoded_u, available_u)
        S = np.array([[float((encoded_q[m] @ encoded_u[m].T).toarray()[0, 0]) for m in MODALITIES]], dtype=np.float32)
        A = available_q & available_u
        S[~A] = 0.0
        neural = model.static.neural.pair_logits(z_q, z_u, S, A)
        parts = model.static.combine(neural, S, A)
        out.append({"id": user["id"], "name": user["name"], "similarity": round(float(parts["static_logit"][0]), 4)})
    return sorted(out, key=lambda x: -x["similarity"])


def print_results(name: str, results: list[dict[str, Any]], scored: dict[str, Any], hide: Sequence[str], rank_by: str) -> None:
    print(f"\nNearest nodes to: {name}  (ranked by {rank_by})")
    print(f"Modalities used: {', '.join(scored['present']) or 'none'}" + (f" (hidden: {', '.join(hide)})" if hide else ""))
    for item in results:
        print(f"\n{item['rank']:>2}. {item['name']} ({item['id']}, {item['level']}) -- {item['category']}")
        print(
            f"    similarity above {100 * item['similarity_percentile']:.2f}% of random pairs; "
            f"forecast above {100 * item['forecast_percentile']:.2f}%"
        )
        for part in item["static_evidence"]:
            shared = "; ".join(part["shared"]) or "-"
            print(f"    + {part['modality']:<11} sim {part['similarity']:.2f}  neural {part['neural_occlusion']:+.2f}  [{shared}]")
        regulatory = item["regulatory_evidence"]
        notes = []
        if regulatory["shared_drug_targets"]:
            notes.append(f"shared drug targets: {', '.join(regulatory['shared_drug_targets'])}")
        if regulatory["gene_is_target_of_other"]:
            notes.append(f"gene targeted by the other's drugs: {', '.join(regulatory['gene_is_target_of_other'])}")
        if regulatory["designations"][1]:
            notes.append(f"{regulatory['designations'][1]} orphan-designated drugs")
        if notes:
            print(f"    regulatory: {'; '.join(notes)}")
        if item["known_regulatory_relation"]:
            known = item["known_regulatory_relation"]
            print(f"    already linked by designation since {known['year']}: {', '.join(known['drugs'])}")


def add_to_graph(query_id: str, record: dict[str, Any], results: list[dict[str, Any]], present: list[str]) -> None:
    nodes_path, edges_path = GRAPH_DIR / "nodes.csv", GRAPH_DIR / "edges.csv"
    if not nodes_path.is_file() or not edges_path.is_file():
        raise FileNotFoundError("Graph files not found. Run `python v3/build_graph.py` first.")
    nodes = pd.read_csv(nodes_path, dtype=str, keep_default_na=False)
    if query_id in set(nodes["orpha_id"]):
        raise ValueError(f"{query_id} is already in the graph; choose another 'id' in the JSON.")
    edges = pd.read_csv(edges_path, dtype=str, keep_default_na=False)
    new_edges = []
    for item in results:
        regulatory = item["regulatory_evidence"]
        plausible = regulatory["shared_drug_targets"] or regulatory["gene_is_target_of_other"] or item["similarities"].get("gene", 0) > 0
        row = {column: "" for column in edges.columns}
        row.update(
            {
                "source": query_id,
                "target": item["id"],
                "source_name": record["name"],
                "target_name": item["name"],
                "similarity": item["similarity"],
                "similarity_percentile": item["similarity_percentile"],
                "forecast": item["forecast"],
                "forecast_percentile": item["forecast_percentile"],
                "rank_source_to_target": item["rank"],
                "mutual": False,
                "support": "plausible" if plausible else "novel",
                "static_evidence": " | ".join(
                    f"{e['modality']} (sim {e['similarity']:.2f}, neural {e['neural_occlusion']:+.2f}): {'; '.join(e['shared']) or '-'}"
                    for e in item["static_evidence"]
                ),
                **{f"sim_{m}": v for m, v in item["similarities"].items()},
            }
        )
        new_edges.append(row)
    node = {column: "" for column in nodes.columns}
    node.update(
        {
            "orpha_id": query_id,
            "name": record["name"],
            "level": "user",
            "top_category": "User-added disease",
            "designated_drugs": len(record.get("drugs", {})),
            "regulatory_relations": 0,
            "degree": len(results),
            "n_modalities": len(present),
            "modalities": ";".join(present),
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

    edges["similarity"] = pd.to_numeric(edges["similarity"])
    edges["forecast"] = pd.to_numeric(edges["forecast"])
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
    parser.add_argument("--rank-by", choices=("similarity", "forecast"), default="similarity")
    parser.add_argument("--add", action="store_true", help="Insert the new disease and its edges into the graph files.")
    parser.add_argument("--output", type=Path, help="Write the ranked neighbours as JSON.")
    args = parser.parse_args(argv)

    hide = [m.strip() for m in args.hide.split(",") if m.strip()]
    unknown = sorted(set(hide) - set(MODALITIES))
    if unknown:
        parser.error(f"unknown modalities in --hide: {unknown}")

    model = load_model()
    world = model.world
    oncology = False
    if args.orpha:
        if args.add:
            parser.error("--add is only for new diseases given with --json")
        query_id = normalize_orpha_id(args.orpha)
        if query_id not in world.index:
            parser.error(f"{query_id} is not a v3 node")
        record = {**world.bundle.records[query_id], "drugs": {}}
        oncology = bool(world.oncology[world.index[query_id]])
        warnings: list[str] = []
    else:
        document = json.loads(args.json.read_text(encoding="utf-8"))
        record, warnings = record_from_user_input(document, world.bundle.knowledge)
        if not record["name"]:
            parser.error("the JSON needs a 'name'")
        query_id = str(document.get("id") or f"USER:{slug(record['name'])}")
        oncology = bool(document.get("oncology", False))
    for warning in warnings:
        print(f"warning: {warning}")

    results, scored = place(model, record, query_id, args.top, hide, args.rank_by, oncology)
    print_results(record["name"], results, scored, hide, args.rank_by)
    users = [u for u in load_user_diseases() if u["id"] != query_id]
    user_matches = compare_user_diseases(model, record, users, hide)
    if user_matches:
        print("\nPreviously added user diseases (similarity logit):")
        for match in user_matches[:5]:
            print(f"    {match['name']} ({match['id']}): {match['similarity']:+.2f}")
    if args.orpha:
        relatives = world.regulatory.relations
        known = relatives[(relatives["a"] == query_id) | (relatives["b"] == query_id)]
        others = (set(known["a"]) | set(known["b"])) - {query_id}
        recovered = sum(1 for item in results if item["id"] in others)
        print(f"\n{recovered} of the top {args.top} already share an orphan-designated drug with {query_id} "
              f"({len(others)} such nodes in total).")
    if args.output:
        args.output.write_text(
            json.dumps({"query": query_id, "name": record["name"], "neighbors": results, "user_diseases": user_matches}, indent=2, default=str),
            encoding="utf-8",
        )
    if args.add:
        add_to_graph(query_id, record, results, scored["present"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
