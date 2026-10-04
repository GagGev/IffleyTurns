"""Compare the frozen v3 production model with the acquired literature pairs.

External validation only: no v3 model reads the literature data.  The
literature scores (automated 0-4 ordinal scores per dimension, from
``.data/literature_acquisition/papers.sqlite``) are aggregated exactly as in
``v2/evaluate_literature.py``, and the same summaries are computed, so v2 and
v3 can be compared on identical pairs.

Usage:  python v3/evaluate_literature.py
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from common import GRAPH_DIR, PROJECT_ROOT, V3_DATA_DIR, configure_stdout, pair_key, timed, write_json
from modalities import MODALITIES
from production import load_model

DEFAULT_DATABASE = PROJECT_ROOT / ".data" / "literature_acquisition" / "papers.sqlite"
OUTPUT_DIR = V3_DATA_DIR / "literature_validation"
V2_RESULTS = PROJECT_ROOT / "v2" / ".data" / "literature_validation" / "pair_results.csv"
CHUNK = 64
DIMENSIONS = (
    "clinical_phenotype",
    "genetic_etiology",
    "molecular_mechanism",
    "natural_history",
    "therapeutic_similarity",
)
ALIGNED_MODALITIES = {
    "clinical_phenotype": ("phenotype",),
    "genetic_etiology": ("gene", "inheritance"),
    "molecular_mechanism": ("pathway",),
    "natural_history": ("onset",),
}


def load_literature(database: Path) -> list[dict[str, Any]]:
    """Unique literature pairs with paper-averaged scores in [0, 1] (as in v2)."""

    db = sqlite3.connect(database)
    db.row_factory = sqlite3.Row
    try:
        metadata = {
            row["paper_pair_id"]: dict(row)
            for row in db.execute(
                """SELECT pp.paper_pair_id, pp.paper_id, pp.orpha_id_a, pp.orpha_id_b,
                a.preferred_name AS disease_a, b.preferred_name AS disease_b, p.pmid
                FROM paper_pairs pp
                JOIN papers p ON p.paper_id = pp.paper_id
                JOIN orpha_entities a ON a.orpha_id = pp.orpha_id_a
                JOIN orpha_entities b ON b.orpha_id = pp.orpha_id_b"""
            )
        }
        scores: dict[int, list[sqlite3.Row]] = defaultdict(list)
        for row in db.execute(
            """SELECT paper_pair_id, dimension, ordinal_score, extraction_confidence
            FROM dimension_scores WHERE ordinal_score IS NOT NULL"""
        ):
            scores[row["paper_pair_id"]].append(row)
    finally:
        db.close()

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for paper_pair_id, paper in metadata.items():
        observations = scores.get(paper_pair_id, [])
        if not observations:
            continue
        key = pair_key(paper["orpha_id_a"], paper["orpha_id_b"])
        names = {paper["orpha_id_a"]: paper["disease_a"], paper["orpha_id_b"]: paper["disease_b"]}
        item = grouped.setdefault(
            key, {"a": key[0], "b": key[1], "name_a": names[key[0]], "name_b": names[key[1]], "papers": [], "dimensions": defaultdict(list)}
        )
        weight = sum(row["extraction_confidence"] for row in observations)
        item["papers"].append(sum(row["ordinal_score"] / 4.0 * row["extraction_confidence"] for row in observations) / weight)
        for row in observations:
            item["dimensions"][row["dimension"]].append((row["ordinal_score"] / 4.0, row["extraction_confidence"]))

    rows = []
    for item in grouped.values():
        record = {
            "orpha_id_a": item["a"],
            "orpha_id_b": item["b"],
            "disease_a": item["name_a"],
            "disease_b": item["name_b"],
            "literature_similarity": float(np.mean(item["papers"])),
            "paper_observations": len(item["papers"]),
        }
        for dimension in DIMENSIONS:
            values = item["dimensions"].get(dimension, [])
            record[f"literature_{dimension}"] = (
                sum(s * w for s, w in values) / sum(w for _, w in values) if values else np.nan
            )
        rows.append(record)
    return sorted(rows, key=lambda r: (r["orpha_id_a"], r["orpha_id_b"]))


def score_pairs(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """v3 similarity, forecast and within-disease ranks for every evaluable pair.

    Ranks are taken among each disease's candidates (all nodes except itself and
    its Orphanet ancestors/descendants), from both sides; the better one is kept,
    matching the top-k graph where an edge exists if either side ranks the other."""

    model = load_model()
    world = model.world
    index = world.index
    evaluable = [r for r in rows if r["orpha_id_a"] in index and r["orpha_id_b"] in index]
    pairs = np.array([(index[r["orpha_id_a"]], index[r["orpha_id_b"]]) for r in evaluable])
    partners: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for k, (a, b) in enumerate(pairs):
        partners[a].append((k, b))
        partners[b].append((k, a))

    n = len(pairs)
    similarity = np.full(n, np.nan, dtype=np.float32)
    forecast = np.full(n, np.nan, dtype=np.float32)
    sims = np.full((n, len(MODALITIES)), np.nan, dtype=np.float32)
    best_rank = np.full(n, np.inf)
    best_forecast_rank = np.full(n, np.inf)
    nodes = np.array(sorted(partners))
    with timed(f"Scoring {len(nodes):,} literature diseases against all {world.n:,} nodes"):
        for start in range(0, len(nodes), CHUNK):
            rows_ = nodes[start : start + CHUNK]
            block = model.score_nodes(rows_)
            excluded = world.nested[rows_].toarray() > 0
            excluded[np.arange(len(rows_)), rows_] = True
            static = np.where(excluded, -np.inf, block["static_logit"])
            fc = np.where(excluded, -np.inf, block["forecast"])
            for i, node in enumerate(rows_):
                for k, other in partners[node]:
                    similarity[k] = block["static_logit"][i, other]
                    forecast[k] = block["forecast"][i, other]
                    sims[k] = [block[f"sim_{m}"][i, other] for m in MODALITIES]
                    if not excluded[i, other]:
                        best_rank[k] = min(best_rank[k], 1 + (static[i] > static[i, other]).sum())
                        best_forecast_rank[k] = min(best_forecast_rank[k], 1 + (fc[i] > fc[i, other]).sum())

    edges = pd.read_csv(GRAPH_DIR / "edges.csv", usecols=["source", "target", "mutual"], low_memory=False)
    edge_lookup = {pair_key(s, t): bool(m) for s, t, m in zip(edges["source"], edges["target"], edges["mutual"])}
    relations = {pair_key(r.a, r.b): int(r.year) for r in world.regulatory.relations.itertuples(index=False)}
    frame = pd.DataFrame(evaluable)
    frame["v3_similarity"] = similarity
    frame["v3_similarity_percentile"] = model.similarity_percentile(similarity)
    frame["v3_forecast"] = forecast
    frame["v3_forecast_percentile"] = model.forecast_percentile(forecast)
    frame["v3_best_rank"] = np.where(np.isfinite(best_rank), best_rank, np.nan)
    frame["v3_best_forecast_rank"] = np.where(np.isfinite(best_forecast_rank), best_forecast_rank, np.nan)
    keys = [pair_key(a, b) for a, b in zip(frame["orpha_id_a"], frame["orpha_id_b"])]
    frame["nested"] = [bool(world.nested[index[a], index[b]]) for a, b in keys]
    frame["in_top10_graph"] = [k in edge_lookup for k in keys]
    frame["graph_mutual"] = [edge_lookup.get(k, False) for k in keys]
    frame["regulatory_relation_year"] = [relations.get(k) for k in keys]
    for j, m in enumerate(MODALITIES):
        frame[f"v3_sim_{m}"] = sims[:, j]
    for dimension, selected in ALIGNED_MODALITIES.items():
        frame[f"v3_aligned_{dimension}"] = frame[[f"v3_sim_{m}" for m in selected]].mean(axis=1, skipna=True)
    frame["v3_aligned_therapeutic_similarity"] = frame["v3_forecast"]
    return frame


def spearman(x: Iterable[float], y: Iterable[float]) -> dict[str, Any]:
    left, right = np.asarray(list(x), dtype=float), np.asarray(list(y), dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3 or np.ptp(left[valid]) == 0 or np.ptp(right[valid]) == 0:
        return {"n": int(valid.sum()), "spearman": None}
    return {"n": int(valid.sum()), "spearman": float(spearmanr(left[valid], right[valid]).statistic)}


def high_low_auc(literature: pd.Series, score: pd.Series) -> float | None:
    mask = ((literature > 0.50) | (literature <= 0.25)) & score.notna()
    labels = (literature[mask] > 0.50).astype(int)
    return float(roc_auc_score(labels, score[mask])) if labels.nunique() == 2 else None


def retrieval(frame: pd.DataFrame, percentile: str, top10: str, density: float) -> dict[str, Any]:
    rate = float(frame[top10].mean())
    return {
        "pairs": int(len(frame)),
        "top_decile_rate": float((frame[percentile] >= 0.90).mean()),
        "top_one_percent_rate": float((frame[percentile] >= 0.99).mean()),
        "top10_graph_rate": rate,
        "top10_graph_enrichment": rate / density,
        "spearman_vs_literature": spearman(frame["literature_similarity"], frame[percentile])["spearman"],
        "high_vs_low_auc": high_low_auc(frame["literature_similarity"], frame[percentile]),
    }


def edge_density(graph_summary: Path) -> float:
    graph = json.loads(graph_summary.read_text(encoding="utf-8"))
    return graph["edges"] / (graph["nodes"] * (graph["nodes"] - 1) / 2)


def summarize(frame: pd.DataFrame, total_pairs: int) -> dict[str, Any]:
    graph = json.loads((GRAPH_DIR / "graph_summary.json").read_text(encoding="utf-8"))
    n_nodes = graph["nodes"]
    density = edge_density(GRAPH_DIR / "graph_summary.json")
    ranked = frame[~frame["nested"]]
    summary: dict[str, Any] = {
        "coverage": {
            "literature_pairs": total_pairs,
            "v3_evaluable_pairs": int(len(frame)),
            "nested_pairs": int(frame["nested"].sum()),
        },
        "similarity": retrieval(frame, "v3_similarity_percentile", "in_top10_graph", density),
        "similarity_within_disease_rank": {
            f"top_{k}_rate": float((ranked["v3_best_rank"] <= k).mean()) for k in (10, 50, 100)
        },
        "forecast": {
            "top_decile_rate": float((frame["v3_forecast_percentile"] >= 0.90).mean()),
            "top_one_percent_rate": float((frame["v3_forecast_percentile"] >= 0.99).mean()),
            "spearman_vs_literature": spearman(frame["literature_similarity"], frame["v3_forecast"])["spearman"],
            "high_vs_low_auc": high_low_auc(frame["literature_similarity"], frame["v3_forecast"]),
            **{f"within_disease_top_{k}_rate": float((ranked["v3_best_forecast_rank"] <= k).mean()) for k in (10, 50)},
        },
        "regulatory_overlap": {
            "literature_pairs_with_regulatory_relation": int(frame["regulatory_relation_year"].notna().sum()),
            "rate": float(frame["regulatory_relation_year"].notna().mean()),
            "random_pair_rate": graph["regulatory_relations_total"] / (n_nodes * (n_nodes - 1) / 2),
        },
        "graph_density": density,
        "by_dimension": {},
    }
    for dimension in DIMENSIONS:
        literature = frame[f"literature_{dimension}"]
        summary["by_dimension"][dimension] = {
            "aligned": list(ALIGNED_MODALITIES.get(dimension, ("forecast",))),
            "vs_aligned": spearman(literature, frame[f"v3_aligned_{dimension}"]),
            "vs_similarity": spearman(literature, frame["v3_similarity"]),
        }

    if V2_RESULTS.is_file():
        v2 = pd.read_csv(V2_RESULTS)
        columns = ["orpha_id_a", "orpha_id_b", "v2_percentile", "in_top10_graph", "v2_sim_drug", "v2_sim_drug_target"]
        both = frame.merge(v2[columns], on=["orpha_id_a", "orpha_id_b"], suffixes=("", "_v2"))
        v2_density = edge_density(V2_RESULTS.parent.parent / "graph" / "graph_summary.json")
        summary["same_pairs_v2_vs_v3"] = {
            "pairs": int(len(both)),
            "v2": retrieval(both, "v2_percentile", "in_top10_graph_v2", v2_density),
            "v3": retrieval(both, "v3_similarity_percentile", "in_top10_graph", density),
        }
        drug = both["v2_sim_drug"].notna() | both["v2_sim_drug_target"].notna()
        literature = both["literature_similarity"]
        strata = {
            "all": both,
            "not nested": both[~both["nested"]],
            "v2 had drug data for both": both[drug],
            "v2 had no drug data": both[~drug],
            "already a v3 regulatory relation": both[both["regulatory_relation_year"].notna()],
            "literature low (<=0.25)": both[literature <= 0.25],
            "literature moderate": both[(literature > 0.25) & (literature <= 0.50)],
            "literature high (>0.50)": both[literature > 0.50],
        }
        summary["top_decile_by_stratum"] = {
            name: {
                "pairs": int(len(part)),
                "v2_similarity": float((part["v2_percentile"] >= 0.90).mean()),
                "v3_similarity": float((part["v3_similarity_percentile"] >= 0.90).mean()),
                "v3_forecast": float((part["v3_forecast_percentile"] >= 0.90).mean()),
            }
            for name, part in strata.items()
        }
    return summary


def write_report(summary: dict[str, Any], path: Path) -> None:
    s, f, r = summary["similarity"], summary["forecast"], summary["regulatory_overlap"]
    lines = [
        "# v3 versus the acquired literature pairs",
        "",
        f"{summary['coverage']['v3_evaluable_pairs']:,} of {summary['coverage']['literature_pairs']:,} literature pairs are v3 nodes "
        f"({summary['coverage']['nested_pairs']} are a group and its member). External validation only; nothing was fitted on these pairs.",
        "",
        "## Similarity (drug-free graph score)",
        "",
        f"- Top decile of random pairs: {s['top_decile_rate']:.1%}; top 1%: {s['top_one_percent_rate']:.1%}.",
        f"- Direct top-10 graph edges: {s['top10_graph_rate']:.1%} ({s['top10_graph_enrichment']:.0f}x a random pair).",
        f"- Within-disease rank (best side, non-nested pairs): top 10 {summary['similarity_within_disease_rank']['top_10_rate']:.1%}, "
        f"top 50 {summary['similarity_within_disease_rank']['top_50_rate']:.1%}.",
        f"- Strength agreement: Spearman {s['spearman_vs_literature']:.3f}; high (>0.5) vs low (<=0.25) AUC {s['high_vs_low_auc']:.3f}.",
        "",
        "## Forecast (future shared orphan designation)",
        "",
        f"- Top decile: {f['top_decile_rate']:.1%}; top 1%: {f['top_one_percent_rate']:.1%}; Spearman {f['spearman_vs_literature']:.3f}.",
        f"- Literature pairs already linked by designations: {r['literature_pairs_with_regulatory_relation']} ({r['rate']:.2%}, "
        f"random pair {r['random_pair_rate']:.4%}).",
        "",
        "## By literature dimension (Spearman)",
        "",
        "| dimension | v3 aligned score | n | vs aligned | vs overall similarity |",
        "|---|---|---|---|---|",
    ]
    for dimension, d in summary["by_dimension"].items():
        aligned, overall = d["vs_aligned"], d["vs_similarity"]
        fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
        lines.append(f"| {dimension} | {', '.join(d['aligned'])} | {aligned['n']} | {fmt(aligned['spearman'])} | {fmt(overall['spearman'])} |")
    same = summary.get("same_pairs_v2_vs_v3")
    if same:
        lines += ["", f"## v2 versus v3 on the same {same['pairs']:,} pairs", "", "| metric | v2 | v3 |", "|---|---|---|"]
        for key in ("top_decile_rate", "top_one_percent_rate", "top10_graph_rate", "top10_graph_enrichment", "spearman_vs_literature", "high_vs_low_auc"):
            lines.append(f"| {key} | {same['v2'][key]:.3f} | {same['v3'][key]:.3f} |")
        lines += [
            "",
            "Share of literature pairs in the top decile of random pairs:",
            "",
            "| stratum | pairs | v2 similarity | v3 similarity | v3 forecast |",
            "|---|---|---|---|---|",
        ]
        for name, row in summary["top_decile_by_stratum"].items():
            lines.append(f"| {name} | {row['pairs']:,} | {row['v2_similarity']:.1%} | {row['v3_similarity']:.1%} | {row['v3_forecast']:.1%} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    configure_stdout()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    args = parser.parse_args()
    literature = load_literature(args.database)
    frame = score_pairs(literature)
    summary = summarize(frame, len(literature))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_DIR / "pair_results.csv", index=False)
    write_json(OUTPUT_DIR / "summary.json", summary)
    write_report(summary, OUTPUT_DIR / "REPORT.md")
    print((OUTPUT_DIR / "REPORT.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
