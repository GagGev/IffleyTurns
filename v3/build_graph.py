"""Build the production v3 rare-disease graph.

Refits the production model on all data, links every node (disease or
Orphanet group) to its ``k`` most similar nodes by the drug-free similarity,
and annotates each edge with:

* the forecast score that the pair will be linked by a future orphan
  designation (and its percentile among random pairs);
* whether the pair is already linked by designations (with the drugs);
* explanations: shared phenotypes/genes/words per modality, the neural
  model's occlusion drop per modality, shared drug targets, genes that are
  targets of the other disease's drugs, and common regulatory relatives.

It also writes ``forecast_pairs.csv``: the pairs not yet linked that the
model ranks most likely to share a designated drug next.

Run after ``run_evaluation.py``:  python v3/build_graph.py [--refit]
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from typing import Any, Optional, Sequence

import networkx as nx
import numpy as np
import pandas as pd

from common import GRAPH_DIR, configure_stdout, pair_key, timed, write_json
from modalities import MODALITIES
from production import MODEL_PATH, ProductionModel, fit_production, load_model, save_model
from world import load_world

CHUNK = 64
FORECAST_PAIRS = 2000
EXPLAINED_FORECASTS = 300
EXAMPLE_DISEASES = (
    "Marfan syndrome",
    "Cystic fibrosis",
    "Duchenne muscular dystrophy",
    "Gaucher disease",
    "Fabry disease",
    "Phenylketonuria",
    "Huntington disease",
    "Amyotrophic lateral sclerosis",
    "Wilson disease",
    "Hemophilia A",
    "Sickle cell anemia",
    "Skeletal Ewing sarcoma",
    "Neurofibromatosis type 1",
    "Tuberous sclerosis complex",
    "Myasthenia gravis",
    "Idiopathic pulmonary fibrosis",
)


def known_relations(model: ProductionModel) -> dict[tuple[str, str], dict[str, Any]]:
    out = {}
    for row in model.world.regulatory.relations.itertuples(index=False):
        out[pair_key(row.a, row.b)] = {
            "year": int(row.year),
            "drugs": "; ".join(model.drug_label(d) for d in row.drugs[:4]),
            "approved_both": bool(row.approved_both),
        }
    return out


def score_all(model: ProductionModel, k: int) -> dict[str, Any]:
    world = model.world
    n, m = world.n, len(MODALITIES)
    nested = world.nested.tocsr()
    warm = world.snapshot(None).warm
    adjacency = model.features.adjacency
    index = np.zeros((n, k), dtype=np.int64)
    similarity = np.zeros((n, k), dtype=np.float32)
    forecast = np.zeros((n, k), dtype=np.float32)
    S_top = np.zeros((n, k, m), dtype=np.float32)
    A_top = np.zeros((n, k, m), dtype=bool)
    candidates: list[tuple[float, int, int]] = []
    for start in range(0, n, CHUNK):
        rows = np.arange(start, min(start + CHUNK, n))
        block = model.score_nodes(rows)
        scores = block["static_logit"].copy()
        excluded = nested[rows].toarray() > 0
        excluded[np.arange(len(rows)), rows] = True
        scores[excluded] = -np.inf
        part = np.argpartition(-scores, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1)
        best = np.take_along_axis(part, order, axis=1)
        local = np.arange(len(rows))[:, None]
        index[rows] = best
        similarity[rows] = scores[local, best]
        forecast[rows] = block["forecast"][local, best]
        for j, name in enumerate(MODALITIES):
            values = block[f"sim_{name}"][local, best]
            A_top[rows, :, j] = ~np.isnan(values)
            S_top[rows, :, j] = np.nan_to_num(values)
        related = (adjacency[rows].toarray() > 0) | excluded
        for i, row in enumerate(rows):
            if not warm[row]:
                continue
            values = np.where(related[i], -np.inf, block["forecast"][i])
            top = np.argpartition(-values, FORECAST_PAIRS)[:FORECAST_PAIRS]
            candidates.extend((float(values[j]), int(row), int(j)) for j in top if np.isfinite(values[j]))
        if (start // CHUNK) % 25 == 0:
            print(f"[v3]   scored {min(start + CHUNK, n):,}/{n:,} nodes", flush=True)
    best_pairs: dict[tuple[int, int], float] = {}
    for value, a, b in candidates:
        key = (min(a, b), max(a, b))
        best_pairs[key] = max(value, best_pairs.get(key, -np.inf))
    ranked = sorted(best_pairs.items(), key=lambda item: -item[1])[:FORECAST_PAIRS]
    return {"index": index, "similarity": similarity, "forecast": forecast, "S": S_top, "A": A_top, "forecast_pairs": ranked}


def support_level(model: ProductionModel, a: int, b: int, known: Optional[dict[str, Any]], regulatory: dict[str, Any]) -> str:
    if known:
        return "regulatory"
    world = model.world
    ra, rb = world.bundle.records[world.ids[a]], world.bundle.records[world.ids[b]]
    parents_a, parents_b = set(ra.get("ontology_parents", ())), set(rb.get("ontology_parents", ()))
    ontology = parents_a & parents_b or world.ids[a] in parents_b or world.ids[b] in parents_a
    if ontology or set(ra.get("genes", {})) & set(rb.get("genes", {})) or regulatory["shared_drug_targets"] or regulatory["gene_is_target_of_other"]:
        return "plausible"
    return "novel"


def format_static(evidence: list[dict[str, Any]]) -> str:
    return " | ".join(
        f"{e['modality']} (sim {e['similarity']:.2f}, neural {e['neural_occlusion']:+.2f}): {'; '.join(e['shared']) or '-'}"
        for e in evidence
    )


def format_regulatory(evidence: dict[str, Any]) -> str:
    parts = [f"designations {evidence['designations'][0]}/{evidence['designations'][1]}"]
    for key, title in (
        ("shared_designated_drugs", "shared drugs"),
        ("shared_drug_targets", "shared drug targets"),
        ("gene_is_target_of_other", "gene targeted by the other's drugs"),
        ("common_regulatory_relatives", "common relatives"),
    ):
        if evidence.get(key):
            parts.append(f"{title}: {'; '.join(evidence[key])}")
    return " | ".join(parts)


def build_edges(model: ProductionModel, scored: dict[str, Any], k: int) -> pd.DataFrame:
    world = model.world
    ids = world.ids
    index, S_top, A_top = scored["index"], scored["S"], scored["A"]
    directed = {(i, int(index[i, r])): r + 1 for i in range(world.n) for r in range(k)}
    pairs, slots = [], []
    seen: set[tuple[int, int]] = set()
    for i in range(world.n):
        for r in range(k):
            j = int(index[i, r])
            key = (min(i, j), max(i, j))
            if key not in seen:
                seen.add(key)
                pairs.append(key)
                slots.append((i, r))
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    S = np.stack([S_top[i, r] for i, r in slots])
    A = np.stack([A_top[i, r] for i, r in slots])
    with timed(f"Neural occlusion for {len(pairs):,} edges"):
        occlusion = model.neural_occlusion(a, b, S, A)
    similarity = np.array([scored["similarity"][i, r] for i, r in slots])
    forecast = np.array([scored["forecast"][i, r] for i, r in slots])
    sim_pct = model.similarity_percentile(similarity)
    fc_pct = model.forecast_percentile(forecast)
    known = known_relations(model)
    rows = []
    with timed("Explaining edges"):
        for e, ((x, y), (i, r)) in enumerate(zip(pairs, slots)):
            query_rows = {m: world.matrices[m][[x]] for m in MODALITIES}
            static = model.static_evidence(query_rows, 0, y, S[e], A[e], occlusion[e])
            regulatory = model.regulatory_evidence(x, y)
            relation = known.get(pair_key(ids[x], ids[y]))
            rows.append(
                {
                    "source": ids[x],
                    "target": ids[y],
                    "source_name": world.name(ids[x]),
                    "target_name": world.name(ids[y]),
                    "similarity": round(float(similarity[e]), 4),
                    "similarity_percentile": round(float(sim_pct[e]), 6),
                    "forecast": round(float(forecast[e]), 4),
                    "forecast_percentile": round(float(fc_pct[e]), 6),
                    "rank_source_to_target": directed.get((x, y), ""),
                    "rank_target_to_source": directed.get((y, x), ""),
                    "mutual": bool(directed.get((x, y)) and directed.get((y, x))),
                    "support": support_level(model, x, y, relation, regulatory),
                    "regulatory_relation_year": relation["year"] if relation else "",
                    "regulatory_relation_drugs": relation["drugs"] if relation else "",
                    "static_evidence": format_static(static),
                    "regulatory_evidence": format_regulatory(regulatory),
                    **{f"sim_{m}": round(float(S[e, c]), 4) if A[e, c] else "" for c, m in enumerate(MODALITIES)},
                    **{f"neural_{m}": round(float(occlusion[e, c]), 4) for c, m in enumerate(MODALITIES)},
                }
            )
    return pd.DataFrame(rows).sort_values("similarity", ascending=False).reset_index(drop=True)


def build_forecasts(model: ProductionModel, scored: dict[str, Any]) -> pd.DataFrame:
    world = model.world
    ids = world.ids
    ranked = scored["forecast_pairs"]
    values = np.array([v for _, v in ranked])
    percentiles = model.forecast_percentile(values)
    drugs = world.snapshot(None).drugs
    rows = []
    for rank, (((a, b), value), pct) in enumerate(zip(ranked, percentiles), start=1):
        joint = set(drugs.get(ids[a], {})) & set(drugs.get(ids[b], {}))
        record = {
            "rank": rank,
            "a": ids[a],
            "b": ids[b],
            "a_name": world.name(ids[a]),
            "b_name": world.name(ids[b]),
            "forecast": round(float(value), 4),
            "forecast_percentile": round(float(pct), 6),
            "a_category": world.top_category[a],
            "b_category": world.top_category[b],
            # Only possible through one joint designation record, which is not a relation.
            "joint_designation_drugs": "; ".join(sorted(model.drug_label(d) for d in joint)[:3]),
        }
        if rank <= EXPLAINED_FORECASTS:
            record["regulatory_evidence"] = format_regulatory(model.regulatory_evidence(a, b))
            S, A = model._sim.pairs(np.array([a]), np.array([b]))
            query_rows = {m: world.matrices[m][[a]] for m in MODALITIES}
            record["static_evidence"] = format_static(model.static_evidence(query_rows, 0, b, S[0], A[0]))
        rows.append(record)
    return pd.DataFrame(rows)


def build_nodes(model: ProductionModel, edges: pd.DataFrame) -> pd.DataFrame:
    world = model.world
    degree = Counter(edges["source"]) + Counter(edges["target"])
    snapshot = world.snapshot(None)
    regulatory_degree = snapshot.degree
    rows = []
    for i, node in enumerate(world.ids):
        meta = world.bundle.meta.loc[node]
        rows.append(
            {
                "orpha_id": node,
                "name": world.name(node),
                "level": meta["level"],
                "top_category": meta["top_category"],
                "designated_drugs": int(snapshot.designations[i]),
                "regulatory_relations": int(regulatory_degree[i]),
                "degree": degree.get(node, 0),
                "n_modalities": int(world.available[i].sum()),
                "modalities": ";".join(m for m, flag in zip(MODALITIES, world.available[i]) if flag),
                "orphanet_url": f"https://www.orpha.net/en/disease/detail/{node.split(':')[1]}",
                "source_type": "orphanet",
            }
        )
    return pd.DataFrame(rows)


def write_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    for row in nodes.itertuples(index=False):
        graph.add_node(row.orpha_id, name=row.name, top_category=row.top_category, level=row.level, source_type=row.source_type)
    for row in edges.itertuples(index=False):
        graph.add_edge(
            row.source,
            row.target,
            similarity=float(row.similarity),
            forecast=float(row.forecast),
            mutual=bool(row.mutual),
            support=row.support,
        )
    nx.write_graphml(graph, GRAPH_DIR / "rare_disease_graph.graphml")
    return graph


def review_sheet(model: ProductionModel, edges: pd.DataFrame, top: int = 5) -> list[dict[str, Any]]:
    world = model.world
    by_name = {world.name(d).casefold(): d for d in world.ids}
    neighbors = defaultdict(list)
    for row in edges.itertuples(index=False):
        neighbors[row.source].append((row.target, row))
        neighbors[row.target].append((row.source, row))
    sheet = []
    for name in EXAMPLE_DISEASES:
        disease = by_name.get(name.casefold())
        if disease is None:
            print(f"[v3] review sheet: {name!r} not found; skipped")
            continue
        ranked = sorted(neighbors[disease], key=lambda item: -item[1].similarity)[:top]
        for position, (other, row) in enumerate(ranked, start=1):
            sheet.append(
                {
                    "query": disease,
                    "query_name": world.name(disease),
                    "rank": position,
                    "neighbor": other,
                    "neighbor_name": world.name(other),
                    "similarity_percentile": row.similarity_percentile,
                    "forecast_percentile": row.forecast_percentile,
                    "support": row.support,
                    "regulatory_relation": f"{row.regulatory_relation_year}: {row.regulatory_relation_drugs}" if row.regulatory_relation_year != "" else "",
                    "static_evidence": row.static_evidence,
                    "regulatory_evidence": row.regulatory_evidence,
                    "manual_status": "",
                }
            )
    return sheet


def write_review_markdown(sheet: list[dict[str, Any]], forecasts: pd.DataFrame) -> None:
    lines = [
        "# v3 review sheet\n",
        "Top-5 most similar nodes (drug-free similarity) for example diseases. `support`: **regulatory** = the pair "
        "already shares an orphan-designated drug; **plausible** = shared Orphanet parent, curated gene, drug target, or "
        "a gene targeted by the other's drugs; **novel** = none of these. `forecast` = percentile of the model's score that "
        "the pair will be linked by a future designation.\n",
    ]
    current = None
    for row in sheet:
        if row["query"] != current:
            current = row["query"]
            lines.append(f"\n## {row['query_name']} ({row['query']})\n")
            lines.append("| # | Neighbour | Similarity pct | Forecast pct | Support | Evidence |")
            lines.append("|---|---|---|---|---|---|")
        evidence = (row["regulatory_relation"] or row["static_evidence"]).replace("|", "/")
        lines.append(
            f"| {row['rank']} | {row['neighbor_name']} ({row['neighbor']}) | {row['similarity_percentile']:.4f} | "
            f"{row['forecast_percentile']:.4f} | {row['support']} | {evidence} |"
        )
    lines.append("\n# Top 25 forecast pairs (not yet linked by two separate designations)\n")
    lines.append(
        "`joint record`: the pair already shares a drug through a single designation that names both diseases, "
        "which does not count as a relation.\n"
    )
    lines.append("| # | Disease A | Disease B | Joint record | Evidence |")
    lines.append("|---|---|---|---|---|")
    for row in forecasts.head(25).itertuples(index=False):
        evidence = str(getattr(row, "regulatory_evidence", "")).replace("|", "/")
        lines.append(f"| {row.rank} | {row.a_name} | {row.b_name} | {row.joint_designation_drugs} | {evidence} |")
    (GRAPH_DIR / "review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_overview(model: ProductionModel, nodes: pd.DataFrame, edges: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    world = model.world
    with timed("Computing 2-D layout (t-SNE of the neural disease embeddings)"):
        z = np.mean([m.z for m in model.static.neural.models], axis=0)
        z = PCA(n_components=50, random_state=0).fit_transform(z)
        xy = TSNE(n_components=2, metric="cosine", init="pca", perplexity=30, random_state=0).fit_transform(z)
    index = world.index
    categories = nodes["top_category"].value_counts()
    shown = list(categories.index[:12])
    palette = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(13, 11))
    mutual = edges[edges["mutual"]]
    segments = [(xy[index[s]], xy[index[t]]) for s, t in zip(mutual["source"], mutual["target"])]
    ax.add_collection(LineCollection(segments, colors="#888888", linewidths=0.15, alpha=0.35))
    other = ~nodes["top_category"].isin(shown).to_numpy()
    ax.scatter(xy[other, 0], xy[other, 1], s=4, color="#cccccc", label=f"Other ({other.sum()})", linewidths=0)
    for c, category in enumerate(shown):
        mask = (nodes["top_category"] == category).to_numpy()
        ax.scatter(xy[mask, 0], xy[mask, 1], s=4, color=palette(c), label=f"{category} ({mask.sum()})", linewidths=0)
    designated = nodes["designated_drugs"].to_numpy() > 0
    ax.scatter(xy[designated, 0], xy[designated, 1], s=14, facecolors="none", edgecolors="black", linewidths=0.4, label=f"Has orphan designations ({designated.sum()})")
    ax.set_title(
        f"v3 rare-disease graph: {len(nodes):,} nodes, {len(mutual):,} mutual top-k edges shown\n"
        "(t-SNE of the neural embeddings; colour = Orphanet top-level category)"
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower left", fontsize=7, markerscale=3, frameon=True)
    fig.tight_layout()
    fig.savefig(GRAPH_DIR / "graph_overview.png", dpi=140)
    plt.close(fig)


def plot_ego(model: ProductionModel, edges: pd.DataFrame, disease_name: str, filename: str, k: int = 12) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    world = model.world
    by_name = {world.name(d).casefold(): d for d in world.ids}
    center = by_name.get(disease_name.casefold())
    if center is None:
        return
    incident = edges[(edges["source"] == center) | (edges["target"] == center)].nlargest(k, "similarity")
    members = {center} | set(incident["source"]) | set(incident["target"])
    local = edges[edges["source"].isin(members) & edges["target"].isin(members)]
    graph = nx.Graph()
    for row in local.itertuples(index=False):
        graph.add_edge(row.source, row.target, similarity=row.similarity, support=row.support, forecast=row.forecast_percentile)
    pos = nx.spring_layout(graph, seed=1, k=0.9)
    colors = {"regulatory": "#2e7d32", "plausible": "#1f77b4", "novel": "#e67e22"}
    fig, ax = plt.subplots(figsize=(12, 9))
    for support, color in colors.items():
        selected = [(u, v) for u, v, d in graph.edges(data=True) if d["support"] == support]
        widths = [0.5 + 3.0 * graph[u][v]["forecast"] for u, v in selected]
        nx.draw_networkx_edges(graph, pos, edgelist=selected, edge_color=color, width=widths, alpha=0.7, ax=ax, label=support)
    nx.draw_networkx_nodes(graph, pos, node_color=["#c0392b" if n == center else "#d6e4f0" for n in graph.nodes], node_size=260, ax=ax)
    labels = {n: (world.name(n)[:38] + "...") if len(world.name(n)) > 40 else world.name(n) for n in graph.nodes}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=7, ax=ax)
    ax.legend(title="edge support (width = forecast percentile)", loc="lower right", fontsize=8)
    ax.set_title(f"Neighbourhood of {disease_name} in the v3 graph (top {k} neighbours and the edges among them)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(GRAPH_DIR / filename, dpi=140)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_stdout()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=10, help="Neighbours kept per node.")
    parser.add_argument("--refit", action="store_true", help="Refit the production model even if one is saved.")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args(argv)

    if args.refit or not MODEL_PATH.is_file():
        model = fit_production(load_world())
        save_model(model)
    else:
        model = load_model()
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    with timed(f"Scoring all {model.world.n:,} nodes against each other"):
        scored = score_all(model, args.k)
    edges = build_edges(model, scored, args.k)
    forecasts = build_forecasts(model, scored)
    nodes = build_nodes(model, edges)
    user_diseases = GRAPH_DIR / "user_diseases.jsonl"
    if user_diseases.is_file():
        user_diseases.unlink()
        print("[v3] the rebuilt graph drops user-added diseases; re-run place_disease.py --add to insert them again")
    edges.to_csv(GRAPH_DIR / "edges.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    nodes.to_csv(GRAPH_DIR / "nodes.csv", index=False)
    forecasts.to_csv(GRAPH_DIR / "forecast_pairs.csv", index=False)
    graph = write_graph(nodes, edges)
    components = sorted((len(c) for c in nx.connected_components(graph)), reverse=True)
    stats = {
        "k": args.k,
        "nodes": int(len(nodes)),
        "groups": int((nodes["level"] == "group").sum()),
        "edges": int(len(edges)),
        "mutual_edges": int(edges["mutual"].sum()),
        "edge_support": edges["support"].value_counts().to_dict(),
        "edges_with_regulatory_relation": int((edges["regulatory_relation_year"] != "").sum()),
        "regulatory_relations_total": int(len(model.world.regulatory.relations)),
        "connected_components": len(components),
        "largest_component": components[0] if components else 0,
        "model": {
            **model.config,
            "end_of_data": model.end_of_data,
            "stacker_importance": model.importance,
            "linear_fusion": model.static.linear.coefficients(),
        },
        "top_forecasts": forecasts.head(10)[["a_name", "b_name", "forecast_percentile"]].to_dict("records"),
    }
    write_json(GRAPH_DIR / "graph_summary.json", stats)
    with timed("Exporting neural embeddings"):
        z = np.mean([m.z for m in model.static.neural.models], axis=0)
        frame = pd.DataFrame(z.astype(np.float32), columns=[f"dim_{i}" for i in range(z.shape[1])])
        frame.insert(0, "name", [model.world.name(d) for d in model.world.ids])
        frame.insert(0, "orpha_id", model.world.ids)
        frame.to_parquet(GRAPH_DIR / "embeddings.parquet", index=False)
    sheet = review_sheet(model, edges)
    pd.DataFrame(sheet).to_csv(GRAPH_DIR / "review_top5.csv", index=False)
    write_review_markdown(sheet, forecasts)
    if not args.skip_figures:
        plot_overview(model, nodes, edges)
        plot_ego(model, edges, "Duchenne muscular dystrophy", "ego_duchenne_muscular_dystrophy.png")
        plot_ego(model, edges, "Cystic fibrosis", "ego_cystic_fibrosis.png")
    print(f"[v3] graph: {stats['nodes']:,} nodes, {stats['edges']:,} edges; support {stats['edge_support']}")
    print(f"[v3] wrote graph outputs to {GRAPH_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
