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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graph-dir", type=Path, default=GRAPH_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-layout", action="store_true", help="Skip the t-SNE layout.")
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

    nodes = []
    for r in node_rows:
        present = set(filter(None, r.get("modalities", "").split(";")))
        mask = sum(1 << i for i, m in enumerate(modalities) if m in present)
        node = {
            "id": r["orpha_id"],
            "name": r["name"],
            "c": category_index.get(r["top_category"], -1),
            "t": r.get("disorder_type", ""),
            "m": mask,
        }
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
