"""Convert the v2 similarity graph into the files the frontend loads.

Reads the outputs of ``v2/build_graph.py`` from ``v2/.data/graph/``:
``nodes.csv``, ``edges.csv``, ``graph_summary.json`` and, for the layout,
``embeddings.parquet``.  Writes to ``frontend/public/data/``:

* ``graph.json``: nodes, edges (score, percentile, support, main modality) and
  a 2-D layout.  Loaded on start-up.
* ``details/NN.json``: per-edge explanations (per-modality similarities and
  contributions, shared features, curated relations), split into shards so a
  panel only fetches what it shows.  The shard of an edge is
  ``edge_shard(edge_id)``, mirrored in ``src/data/source.ts``.

The layout is a t-SNE of v2's fused embedding (as in v2's overview figure)
when pyarrow and scikit-learn are installed; otherwise the browser lays the
graph out itself.

Diseases are also clustered with Louvain community detection on the similarity
graph (edge weight = the fusion score, clipped at 0; needs networkx).  Each node
carries its cluster index (``k``) and ``clusters`` describes every cluster
(size, dominant Orphanet category, distinctive name terms, hub disease), so the
UI can colour the graph by cluster.

Usage (from the repository root, after ``python v2/build_graph.py``):
    python3 frontend/scripts/build_graph_data.py
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRAPH_DIR = PROJECT_ROOT / "v2" / ".data" / "graph"
MODALITIES_SOURCE = PROJECT_ROOT / "v2" / "modalities.py"
OUTPUT_DIR = PROJECT_ROOT / "frontend" / "public" / "data"
SHARDS = 64
CLUSTER_RESOLUTION = 0.5  # Louvain resolution: 0.5 gives ~40 clusters, 1.0 ~60 (v2 graph, k=10)
# Words too common in disease names to describe a cluster.
NAME_STOPWORDS = frozenset(
    """syndrome disease disorder disorders type types with without due deficiency rare familial congenital associated
    related early onset inherited autosomal recessive dominant linked acquired primary secondary other unspecified
    forms form and the for from of non""".split()
)
SUPPORT_LEVELS = ("curated", "plausible", "novel")
RELATION_COLUMNS = {
    "orphanet": "known_orphanet_relation",
    "gene": "known_shared_causal_gene",
    "drug": "known_shared_trial_drug",
}

csv.field_size_limit(sys.maxsize)


def edge_shard(edge_id: str) -> int:
    """djb2 over the ID; the frontend computes the same value."""

    h = 5381
    for ch in edge_id:
        h = ((h * 33) + ord(ch)) & 0xFFFFFFFF
    return h % SHARDS


def modality_descriptions() -> dict[str, str]:
    """Read MODALITY_DESCRIPTIONS from v2/modalities.py without importing it (no numpy/sklearn needed)."""

    tree = ast.parse(MODALITIES_SOURCE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "MODALITY_DESCRIPTIONS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit(f"MODALITY_DESCRIPTIONS not found in {MODALITIES_SOURCE}")


def number(value: str) -> float | None:
    try:
        return float(value) if value not in ("", None) else None
    except ValueError:
        return None


def parse_evidence(text: str, modality_index: dict[str, int]) -> list[list]:
    """``phenotype (+1.23): Seizure; Hypotonia | gene (+0.80): CDKL5`` → [[index, contribution, [shared...]], ...]."""

    out = []
    for part in filter(None, (p.strip() for p in (text or "").split(" | "))):
        head, _, shared = part.partition(": ")
        name, _, contribution = head.partition(" (+")
        if name not in modality_index:
            continue
        out.append([
            modality_index[name],
            number(contribution.rstrip(")")) or 0.0,
            [s for s in shared.split("; ") if s],
        ])
    return out


def layout(ids: list[str], graph_dir: Path) -> dict[str, tuple[float, float]] | None:
    path = graph_dir / "embeddings.parquet"
    if not path.is_file():
        print("No embeddings.parquet; the browser will lay out the graph.")
        return None
    try:
        import numpy as np
        import pyarrow.parquet as pq
        from sklearn.manifold import TSNE
    except ImportError as error:
        print(f"Skipping precomputed layout ({error.name} not installed); the browser will lay out the graph.")
        return None
    table = pq.read_table(path).to_pandas()
    dims = [c for c in table.columns if c.startswith("dim_")][:50]
    vectors = np.array(table[dims].to_numpy(dtype="float32"))  # writable copy
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
    print(f"Computing t-SNE layout for {len(vectors):,} diseases...")
    perplexity = min(30, max(2, (len(vectors) - 1) // 3))
    xy = TSNE(n_components=2, metric="cosine", init="pca", perplexity=perplexity, random_state=0).fit_transform(vectors)
    xy = (xy - xy.mean(axis=0)) / max(float(np.abs(xy).max()), 1e-9) * 1000
    positions = {str(i): (round(float(x), 1), round(float(y), 1)) for i, (x, y) in zip(table["orpha_id"], xy)}
    return {i: positions[i] for i in ids if i in positions}


def short_category(category: str) -> str:
    """"Rare neurologic disease" -> "Neurologic"."""

    text = category.replace("Rare ", "", 1) if category.startswith("Rare ") else category
    text = text.removesuffix(" disease").removesuffix(" disorder")
    return text[:1].upper() + text[1:] if text else "Unclassified"


def cluster_nodes(node_rows: list[dict], edge_rows: list[dict], resolution: float) -> tuple[list[int], list[dict], dict] | None:
    """Louvain communities of the similarity graph.

    Returns (cluster index per node row, cluster descriptions, method summary).  Clusters are numbered by size, so
    cluster 0 ("C1") is the largest.  Returns None when networkx is not installed.
    """

    try:
        import networkx as nx
    except ImportError:
        print("Skipping clustering (networkx not installed); the graph will not be coloured by cluster.")
        return None
    ids = [r["orpha_id"] for r in node_rows]
    graph = nx.Graph()
    graph.add_nodes_from(ids)
    for r in edge_rows:
        if r["source"] in graph and r["target"] in graph:
            weight = max(number(r["score"]) or 0.0, 0.0) + 0.05
            graph.add_edge(r["source"], r["target"], weight=weight)
    communities = nx.community.louvain_communities(graph, weight="weight", resolution=resolution, seed=0)
    communities = sorted(communities, key=lambda c: (-len(c), min(c)))
    modularity = nx.community.modularity(graph, communities, weight="weight")
    label_of = {d: i for i, members in enumerate(communities) for d in members}

    names = {r["orpha_id"]: r["name"] for r in node_rows}
    categories = {r["orpha_id"]: r["top_category"] or "Unclassified" for r in node_rows}

    def tokens(name: str) -> set[str]:
        words = (w.strip(".,;:()[]'\"").lower() for w in name.replace("/", " ").replace("-", " ").split())
        return {w for w in words if len(w) >= 4 and w.isalpha() and w not in NAME_STOPWORDS}

    document_frequency: Counter = Counter()
    for name in names.values():
        document_frequency.update(tokens(name))
    total = len(names)

    import math

    clusters = []
    for index, members in enumerate(communities):
        member_list = sorted(members)
        category_counts = Counter(categories[d] for d in member_list)
        top_category, top_count = category_counts.most_common(1)[0]
        counts: Counter = Counter()
        for d in member_list:
            counts.update(tokens(names[d]))
        floor = max(3, int(0.04 * len(member_list)))
        scored = sorted(
            ((c / len(member_list)) * math.log(total / document_frequency[w]), w) for w, c in counts.items() if c >= floor
        )
        terms = [w for _, w in reversed(scored)][:3]
        hub = max(member_list, key=lambda d: sum(graph[d][n]["weight"] for n in graph[d] if label_of[n] == index))
        label = short_category(top_category) + (" · " + ", ".join(terms) if terms else "")
        clusters.append(
            {
                "id": f"C{index + 1}",
                "label": label,
                "size": len(member_list),
                "topCategory": top_category,
                "purity": round(top_count / len(member_list), 3),
                "categories": [[c, n] for c, n in category_counts.most_common(3)],
                "terms": terms,
                "hubId": hub,
                "hubName": names[hub],
            }
        )
    method = {
        "method": "Louvain community detection",
        "resolution": resolution,
        "edgeWeight": "fusion score, clipped at 0",
        "modularity": round(modularity, 4),
        "clusters": len(clusters),
    }
    return [label_of[d] for d in ids], clusters, method


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graph-dir", type=Path, default=GRAPH_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-layout", action="store_true", help="Skip the t-SNE layout.")
    parser.add_argument("--cluster-resolution", type=float, default=CLUSTER_RESOLUTION, help="Louvain resolution (higher = more, smaller clusters).")
    parser.add_argument("--no-clusters", action="store_true", help="Skip clustering.")
    args = parser.parse_args(argv)

    nodes_path, edges_path = args.graph_dir / "nodes.csv", args.graph_dir / "edges.csv"
    if not nodes_path.is_file() or not edges_path.is_file():
        print(
            f"v2 graph not found in {args.graph_dir}. Build it first:\n"
            "  python download_databases.py && python generate_features.py\n"
            "  python v2/run_evaluation.py && python v2/build_graph.py",
            file=sys.stderr,
        )
        return 1

    descriptions = modality_descriptions()
    modalities = list(descriptions)
    modality_index = {m: i for i, m in enumerate(modalities)}

    with open(nodes_path, newline="", encoding="utf-8") as f:
        node_rows = list(csv.DictReader(f))
    with open(edges_path, newline="", encoding="utf-8") as f:
        edge_rows = list(csv.DictReader(f))

    categories = sorted({r["top_category"] for r in node_rows if r["top_category"]})
    category_index = {c: i for i, c in enumerate(categories)}
    ids = [r["orpha_id"] for r in node_rows]
    node_index = {d: i for i, d in enumerate(ids)}
    positions = None if args.no_layout else layout(ids, args.graph_dir)
    clustering = None if args.no_clusters else cluster_nodes(node_rows, edge_rows, args.cluster_resolution)

    nodes = []
    for row_index, r in enumerate(node_rows):
        present = set(filter(None, r.get("modalities", "").split(";")))
        mask = sum(1 << i for i, m in enumerate(modalities) if m in present)
        node = {
            "id": r["orpha_id"],
            "name": r["name"],
            "c": category_index.get(r["top_category"], -1),
            "t": r.get("disorder_type", ""),
            "m": mask,
        }
        if clustering:
            node["k"] = clustering[0][row_index]
        if r.get("source_type") and r["source_type"] != "orphanet":
            node["u"] = 1
        if positions and r["orpha_id"] in positions:
            node["x"], node["y"] = positions[r["orpha_id"]]
        nodes.append(node)

    edges = []
    details: list[dict[str, list]] = [{} for _ in range(SHARDS)]
    skipped = 0
    for r in edge_rows:
        s, t = node_index.get(r["source"]), node_index.get(r["target"])
        if s is None or t is None:
            skipped += 1
            continue
        contributions = [number(r.get(f"contrib_{m}", "")) or 0.0 for m in modalities]
        main = max(range(len(modalities)), key=lambda i: contributions[i]) if any(contributions) else -1
        support = SUPPORT_LEVELS.index(r["support"]) if r["support"] in SUPPORT_LEVELS else 2
        mutual = str(r.get("mutual", "")).lower() == "true"
        # [source, target, score, percentile, support, mutual, main modality]
        edges.append([s, t, number(r["score"]) or 0.0, number(r["percentile"]) or 0.0, support, int(mutual), main])

        edge_id = f"{r['source']}|{r['target']}"
        details[edge_shard(edge_id)][edge_id] = {
            "s": [number(r.get(f"sim_{m}", "")) for m in modalities],
            "c": [round(c, 4) for c in contributions],
            "e": parse_evidence(r.get("evidence", ""), modality_index),
            "r": {k: r[col] for k, col in RELATION_COLUMNS.items() if r.get(col)},
            "a": number(r.get("annotation_adjustment", "")) or 0.0,
            "k": [number(r.get("rank_source_to_target", "")), number(r.get("rank_target_to_source", ""))],
        }

    summary_path = args.graph_dir / "graph_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    support_counts = Counter(SUPPORT_LEVELS[e[4]] for e in edges)
    document = {
        "schemaVersion": 2,
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "v2/.data/graph",
        "k": summary.get("k"),
        "modalities": modalities,
        "modalityDescriptions": descriptions,
        "categories": categories,
        "supportLevels": list(SUPPORT_LEVELS),
        "shards": SHARDS,
        "hasLayout": positions is not None,
        "stats": {
            "nodes": len(nodes),
            "edges": len(edges),
            "mutualEdges": sum(e[5] for e in edges),
            "support": dict(support_counts),
        },
        "model": summary.get("model", {}),
        "clusters": clustering[1] if clustering else [],
        "clustering": clustering[2] if clustering else None,
        "nodes": nodes,
        "edges": edges,
    }

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    for stale in ("literature_graph.json", "disease_features.json"):
        (out / stale).unlink(missing_ok=True)
    (out / "graph.json").write_text(json.dumps(document, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    shard_dir = out / "details"
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir()
    for i, shard in enumerate(details):
        (shard_dir / f"{i:02d}.json").write_text(json.dumps(shard, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    size = (out / "graph.json").stat().st_size / 1e6
    print(f"{len(nodes):,} diseases, {len(edges):,} edges ({skipped} skipped) -> {out / 'graph.json'} ({size:.1f} MB)"
          f" + {SHARDS} detail shards")
    if clustering:
        m = clustering[2]
        print(f"{m['clusters']} clusters ({m['method']}, resolution {m['resolution']}, modularity {m['modularity']}); largest:")
        for c in clustering[1][:6]:
            print(f"  {c['id']:>4} {c['size']:5d}  {c['label']}   (hub: {c['hubName']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
