"""Build the production rare-disease similarity graph.

Fits the selected v2 model on the whole catalogue, links every disease to its
``k`` most similar diseases, and explains each edge by the modalities and the
shared features (phenotypes, genes, pathways, drugs, ...) that produced it.
Edges are flagged when a curated relation (Orphanet classification, shared
causal gene, shared trial drug) backs them, and otherwise marked as plausible
or novel hypotheses.

Run after ``run_evaluation.py``:  python v2/build_graph.py
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from typing import Any, Optional, Sequence

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

from benchmarks import OrphanetIndex
from common import GRAPH_DIR, configure_stdout, pair_key, timed, write_json
from data_sources import Bundle, load_bundle
from production import ProductionModel, fit_production

CHUNK = 256
EXAMPLE_DISEASES = (
    "Marfan syndrome",
    "Cystic fibrosis",
    "Duchenne muscular dystrophy",
    "Gaucher disease",
    "Glycogen storage disease due to acid maltase deficiency",
    "Fabry disease",
    "Phenylketonuria",
    "Huntington disease",
    "Rett syndrome",
    "Amyotrophic lateral sclerosis",
    "Wilson disease",
    "Hemophilia A",
    "Sickle cell anemia",
    "Skeletal Ewing sarcoma",
    "Neurofibromatosis type 1",
    "Tuberous sclerosis complex",
    "Fragile X syndrome",
    "Prader-Willi syndrome",
    "Myasthenia gravis",
    "Idiopathic pulmonary fibrosis",
)
RELATION_COLUMNS = {
    "orphanet_siblings": "known_orphanet_relation",
    "shared_causal_gene": "known_shared_causal_gene",
    "shared_drug": "known_shared_trial_drug",
}


def top_neighbors(model: ProductionModel, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    engine = model.engine
    gallery = engine.gallery(model.ids)
    n, n_mod = len(model.ids), len(engine.modalities)
    index = np.zeros((n, k), dtype=np.int64)
    logits = np.zeros((n, k), dtype=np.float32)
    S_top = np.zeros((n, k, n_mod), dtype=np.float32)
    A_top = np.zeros((n, k, n_mod), dtype=bool)
    for start in range(0, n, CHUNK):
        chunk = model.ids[start : start + CHUNK]
        rows = np.arange(len(chunk))
        Qm, Qa = engine.rows(chunk)
        S, A = engine.block(Qm, Qa, gallery)
        scores = model.fusion.score(S, A)
        scores[rows, start + rows] = -np.inf
        part = np.argpartition(-scores, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1)
        best = np.take_along_axis(part, order, axis=1)
        index[start : start + len(chunk)] = best
        logits[start : start + len(chunk)] = np.take_along_axis(scores, best, axis=1)
        S_top[start : start + len(chunk)] = S[rows[:, None], best]
        A_top[start : start + len(chunk)] = A[rows[:, None], best]
    return index, logits, S_top, A_top


def support_level(bundle: Bundle, orphanet: OrphanetIndex, known: dict[str, str], a: str, b: str) -> str:
    if known:
        return "curated"
    ra, rb = bundle.records[a], bundle.records[b]
    if (
        orphanet.ancestors.get(a, frozenset()) & orphanet.ancestors.get(b, frozenset())
        or set(ra["genes"]) & set(rb["genes"])
        or set(ra["drugs"]) & set(rb["drugs"])
    ):
        return "plausible"
    return "novel"


def format_explanation(explanation: list[dict[str, Any]]) -> str:
    parts = []
    for item in explanation:
        shared = "; ".join(item["shared"])
        parts.append(f"{item['modality']} (+{item['contribution']:.2f}): {shared}")
    return " | ".join(parts)


def build_edges(bundle: Bundle, model: ProductionModel, k: int) -> pd.DataFrame:
    with timed(f"Ranking all {len(model.ids):,} diseases against the catalogue (top {k})"):
        index, logits, S_top, A_top = top_neighbors(model, k)
    orphanet = OrphanetIndex(bundle)
    ids = model.ids
    directed: dict[tuple[int, int], int] = {}
    for i in range(len(ids)):
        for r in range(k):
            directed[(i, int(index[i, r]))] = r + 1
    rows = []
    seen: set[tuple[int, int]] = set()
    percentiles = model.percentile(logits)
    with timed("Explaining edges"):
        for i in range(len(ids)):
            query_rows = {m: model.matrices[m][[i]] for m in model.matrices}
            for r in range(k):
                j = int(index[i, r])
                key = (min(i, j), max(i, j))
                if key in seen:
                    continue
                seen.add(key)
                a, b = ids[key[0]], ids[key[1]]
                rank_ab = directed.get((key[0], key[1]))
                rank_ba = directed.get((key[1], key[0]))
                explanation = model.explain(query_rows, 0, j, S_top[i, r], A_top[i, r])
                known = model.known_relations(a, b)
                contributions = model.fusion.similarity_contributions(S_top[i, r])
                rows.append(
                    {
                        "source": a,
                        "target": b,
                        "source_name": model.names[key[0]],
                        "target_name": model.names[key[1]],
                        "score": round(float(logits[i, r]), 4),
                        "percentile": round(float(percentiles[i, r]), 6),
                        "rank_source_to_target": rank_ab or "",
                        "rank_target_to_source": rank_ba or "",
                        "mutual": bool(rank_ab and rank_ba),
                        "support": support_level(bundle, orphanet, known, a, b),
                        **{column: known.get(name, "") for name, column in RELATION_COLUMNS.items()},
                        "top_modalities": "; ".join(
                            f"{e['modality']}:{e['contribution']:.2f}" for e in explanation
                        ),
                        "evidence": format_explanation(explanation),
                        "annotation_adjustment": round(float(model.fusion.availability_adjustment(A_top[i, r])), 4),
                        **{
                            f"sim_{m}": round(float(S_top[i, r, c]), 4) if A_top[i, r, c] else ""
                            for c, m in enumerate(model.engine.modalities)
                        },
                        **{f"contrib_{m}": round(float(contributions[c]), 4) for c, m in enumerate(model.engine.modalities)},
                    }
                )
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def build_nodes(bundle: Bundle, model: ProductionModel, edges: pd.DataFrame) -> pd.DataFrame:
    degree = Counter(edges["source"]) + Counter(edges["target"])
    available = model.engine.available
    rows = []
    for i, disease in enumerate(model.ids):
        meta = bundle.meta.loc[disease]
        rows.append(
            {
                "orpha_id": disease,
                "name": model.names[i],
                "top_category": meta["top_category"],
                "disorder_type": meta["disorder_type"],
                "degree": degree.get(disease, 0),
                "n_modalities": int(available[i].sum()),
                "modalities": ";".join(m for m, flag in zip(model.engine.modalities, available[i]) if flag),
                "orphanet_url": f"https://www.orpha.net/en/disease/detail/{disease.split(':')[1]}",
                "source_type": "orphanet",
            }
        )
    return pd.DataFrame(rows)


def fused_embedding(model: ProductionModel, dimensions: int = 128) -> np.ndarray:
    """Embedding whose inner product reproduces the fusion's similarity term."""

    from sklearn.decomposition import TruncatedSVD

    weights = np.clip(model.fusion.similarity_weights, 0, None)
    blocks = [model.matrices[m] * math.sqrt(float(w)) for m, w in zip(model.engine.modalities, weights) if w > 0]
    fused = sp.hstack(blocks).tocsr()
    svd = TruncatedSVD(n_components=dimensions, random_state=0)
    return svd.fit_transform(fused)


def graph_statistics(nodes: pd.DataFrame, edges: pd.DataFrame, graph: nx.Graph) -> dict[str, Any]:
    components = sorted((len(c) for c in nx.connected_components(graph)), reverse=True)
    degree = nodes["degree"]
    return {
        "nodes": int(len(nodes)),
        "edges": int(len(edges)),
        "mutual_edges": int(edges["mutual"].sum()),
        "edge_support": edges["support"].value_counts().to_dict(),
        "edges_with_known_relation": {
            column: int((edges[column] != "").sum()) for column in RELATION_COLUMNS.values()
        },
        "degree": {"mean": float(degree.mean()), "median": float(degree.median()), "max": int(degree.max())},
        "highest_degree_nodes": nodes.nlargest(5, "degree")[["orpha_id", "name", "degree"]].to_dict("records"),
        "connected_components": len(components),
        "largest_component": components[0] if components else 0,
    }


def review_sheet(bundle: Bundle, model: ProductionModel, edges: pd.DataFrame, top: int = 5) -> list[dict[str, Any]]:
    by_name = {name.casefold(): i for i, name in enumerate(model.names)}
    neighbors = defaultdict(list)
    for row in edges.itertuples(index=False):
        neighbors[row.source].append((row.target, row))
        neighbors[row.target].append((row.source, row))
    sheet = []
    for name in EXAMPLE_DISEASES:
        i = by_name.get(name.casefold())
        if i is None:
            print(f"[v2] review sheet: {name!r} not found in the cohort; skipped")
            continue
        disease = model.ids[i]
        ranked = sorted(neighbors[disease], key=lambda item: -item[1].score)[:top]
        for position, (other, row) in enumerate(ranked, start=1):
            known = "; ".join(
                value for value in (row.known_orphanet_relation, row.known_shared_causal_gene, row.known_shared_trial_drug) if value
            )
            sheet.append(
                {
                    "query": disease,
                    "query_name": model.names[i],
                    "rank": position,
                    "neighbor": other,
                    "neighbor_name": model.names[model.engine.index[other]],
                    "score": row.score,
                    "percentile": row.percentile,
                    "automatic_status": {"curated": "supported", "plausible": "plausible", "novel": "unsupported"}[row.support],
                    "curated_relation": known,
                    "evidence": row.evidence,
                    "manual_status": "",
                }
            )
    return sheet


def plot_overview(model: ProductionModel, nodes: pd.DataFrame, edges: pd.DataFrame, embedding: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from sklearn.manifold import TSNE

    with timed("Computing 2-D layout (t-SNE of the fused embedding)"):
        normalized = embedding[:, :50] / np.maximum(np.linalg.norm(embedding[:, :50], axis=1, keepdims=True), 1e-9)
        xy = TSNE(n_components=2, metric="cosine", init="pca", perplexity=30, random_state=0).fit_transform(normalized)
    index = model.engine.index
    categories = nodes["top_category"].value_counts()
    shown = list(categories.index[:12])
    palette = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(13, 11))
    mutual = edges[edges["mutual"]]
    segments = [(xy[index[s]], xy[index[t]]) for s, t in zip(mutual["source"], mutual["target"])]
    ax.add_collection(LineCollection(segments, colors="#888888", linewidths=0.15, alpha=0.35))
    other = ~nodes["top_category"].isin(shown).to_numpy()
    ax.scatter(xy[other, 0], xy[other, 1], s=4, color="#cccccc", label=f"Other ({other.sum()})", alpha=0.85, linewidths=0)
    for c, category in enumerate(shown):
        mask = (nodes["top_category"] == category).to_numpy()
        ax.scatter(xy[mask, 0], xy[mask, 1], s=4, color=palette(c), label=f"{category} ({mask.sum()})", alpha=0.85, linewidths=0)
    ax.set_title(f"v2 rare-disease similarity graph: {len(nodes):,} diseases, {len(mutual):,} mutual top-k edges shown\n"
                 "(t-SNE of the fused embedding; colour = Orphanet top-level category)")
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

    by_name = {name.casefold(): d for d, name in zip(model.ids, model.names)}
    center = by_name.get(disease_name.casefold())
    if center is None:
        return
    incident = edges[(edges["source"] == center) | (edges["target"] == center)].nlargest(k, "score")
    members = {center} | set(incident["source"]) | set(incident["target"])
    local = edges[edges["source"].isin(members) & edges["target"].isin(members)]
    graph = nx.Graph()
    for row in local.itertuples(index=False):
        graph.add_edge(row.source, row.target, score=row.score, support=row.support)
    pos = nx.spring_layout(graph, seed=1, k=0.9)
    colors = {"curated": "#2e7d32", "plausible": "#1f77b4", "novel": "#e67e22"}
    fig, ax = plt.subplots(figsize=(12, 9))
    for support, color in colors.items():
        selected = [(u, v) for u, v, d in graph.edges(data=True) if d["support"] == support]
        widths = [0.5 + 0.4 * max(graph[u][v]["score"], 0) for u, v in selected]
        nx.draw_networkx_edges(graph, pos, edgelist=selected, edge_color=color, width=widths, alpha=0.7, ax=ax, label=support)
    nx.draw_networkx_nodes(graph, pos, node_color=["#c0392b" if n == center else "#d6e4f0" for n in graph.nodes], node_size=260, ax=ax)
    names = dict(zip(model.ids, model.names))
    labels = {n: (names[n][:38] + "...") if len(names[n]) > 40 else names[n] for n in graph.nodes}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=7, ax=ax)
    ax.legend(title="edge support", loc="lower right", fontsize=8)
    ax.set_title(f"Neighbourhood of {disease_name} in the v2 graph (top {k} neighbours and the edges among them)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(GRAPH_DIR / filename, dpi=140)
    plt.close(fig)


def write_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    for row in nodes.itertuples(index=False):
        graph.add_node(row.orpha_id, name=row.name, top_category=row.top_category, source_type=row.source_type)
    for row in edges.itertuples(index=False):
        graph.add_edge(
            row.source,
            row.target,
            score=float(row.score),
            percentile=float(row.percentile),
            mutual=bool(row.mutual),
            support=row.support,
            top_modalities=row.top_modalities,
        )
    nx.write_graphml(graph, GRAPH_DIR / "rare_disease_graph.graphml")
    return graph


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_stdout()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=10, help="Neighbours kept per disease.")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args(argv)

    bundle = load_bundle()
    model = fit_production(bundle)
    model.save()
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    edges = build_edges(bundle, model, args.k)
    nodes = build_nodes(bundle, model, edges)
    edges.to_csv(GRAPH_DIR / "edges.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    nodes.to_csv(GRAPH_DIR / "nodes.csv", index=False)
    graph = write_graph(nodes, edges)
    stats = graph_statistics(nodes, edges, graph)
    stats["k"] = args.k
    stats["model"] = {"config": model.config, "coefficients": model.fusion.coefficients(), "intercept": model.fusion.intercept}
    write_json(GRAPH_DIR / "graph_summary.json", stats)

    with timed("Exporting fused embedding"):
        embedding = fused_embedding(model)
        frame = pd.DataFrame(embedding.astype(np.float32), columns=[f"dim_{i}" for i in range(embedding.shape[1])])
        frame.insert(0, "name", model.names)
        frame.insert(0, "orpha_id", model.ids)
        frame.to_parquet(GRAPH_DIR / "embeddings.parquet", index=False)

    sheet = review_sheet(bundle, model, edges)
    pd.DataFrame(sheet).to_csv(GRAPH_DIR / "review_top5.csv", index=False)
    write_review_markdown(sheet)
    if not args.skip_figures:
        plot_overview(model, nodes, edges, embedding)
        plot_ego(model, edges, "Marfan syndrome", "ego_marfan_syndrome.png")
        plot_ego(model, edges, "Duchenne muscular dystrophy", "ego_duchenne_muscular_dystrophy.png")
    print(f"[v2] graph: {stats['nodes']:,} nodes, {stats['edges']:,} edges; support {stats['edge_support']}")
    print(f"[v2] wrote graph outputs to {GRAPH_DIR}")
    return 0


def write_review_markdown(sheet: list[dict[str, Any]]) -> None:
    lines = [
        "# Top-5 neighbour review sheet\n",
        "PROJECT.md initial evaluation, steps 6-7. `automatic_status` is a first pass: **supported** = a curated relation "
        "(Orphanet classification, shared Orphanet causal gene, or shared phase>=2 drug) exists; **plausible** = the diseases "
        "share an Orphanet group, a curated gene, or a drug; **unsupported** = none of these (a novel hypothesis to review). "
        "The `manual_status` column in `review_top5.csv` is left for expert review.\n",
    ]
    counts = Counter(row["automatic_status"] for row in sheet)
    lines.append(f"Overall: {dict(counts)} across {len({r['query'] for r in sheet})} example diseases.\n")
    current = None
    for row in sheet:
        if row["query"] != current:
            current = row["query"]
            lines.append(f"\n## {row['query_name']} ({row['query']})\n")
            lines.append("| # | Neighbour | Percentile | Status | Curated relation | Main evidence |")
            lines.append("|---|---|---|---|---|---|")
        evidence = row["evidence"].replace("|", "/")
        lines.append(
            f"| {row['rank']} | {row['neighbor_name']} ({row['neighbor']}) | {row['percentile']:.4f} | {row['automatic_status']} | "
            f"{row['curated_relation'] or '-'} | {evidence} |"
        )
    (GRAPH_DIR / "review_top5.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
