"""Evaluate the frozen v2 production graph against acquired literature scores.

This is an external validation pass: literature evidence is never used to fit
v2.  Automated paper scores are aggregated without converting NA to zero, then
compared with v2 probabilities, random-pair percentiles, top-10 graph edges,
and dimension-aligned v2 modalities.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

from common import V2_DATA_DIR, configure_stdout, pair_key, write_json
from production import load_model


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATABASE = ROOT / ".data/literature_acquisition/papers.sqlite"
DEFAULT_OUTPUT_DIR = V2_DATA_DIR / "literature_validation"
DIMENSIONS = (
    "clinical_phenotype",
    "genetic_etiology",
    "molecular_mechanism",
    "natural_history",
    "therapeutic_similarity",
)
ALIGNED_MODALITIES = {
    "clinical_phenotype": ("phenotype",),
    "genetic_etiology": ("gene", "ot_gene", "inheritance"),
    "molecular_mechanism": ("pathway",),
    "natural_history": ("onset",),
    "therapeutic_similarity": ("drug", "drug_target"),
}


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def correlation(x: Iterable[float], y: Iterable[float]) -> dict[str, Any]:
    left = np.asarray(list(x), dtype=np.float64)
    right = np.asarray(list(y), dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    if len(left) < 3 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return {"n": int(len(left)), "pearson": None, "spearman": None}
    return {
        "n": int(len(left)),
        "pearson": float(pearsonr(left, right).statistic),
        "spearman": float(spearmanr(left, right).statistic),
    }


def load_literature(database: Path) -> list[dict[str, Any]]:
    db = sqlite3.connect(database)
    db.row_factory = sqlite3.Row
    try:
        metadata = {
            row["paper_pair_id"]: dict(row)
            for row in db.execute(
                """SELECT pp.paper_pair_id,pp.paper_id,pp.orpha_id_a,pp.orpha_id_b,
                a.preferred_name AS disease_a,b.preferred_name AS disease_b,
                p.access_status,p.pmid
                FROM paper_pairs pp
                JOIN papers p ON p.paper_id=pp.paper_id
                JOIN orpha_entities a ON a.orpha_id=pp.orpha_id_a
                JOIN orpha_entities b ON b.orpha_id=pp.orpha_id_b"""
            )
        }
        score_rows: dict[int, list[sqlite3.Row]] = defaultdict(list)
        for row in db.execute(
            """SELECT paper_pair_id,dimension,ordinal_score,
            extraction_confidence,evidence_access_status
            FROM dimension_scores WHERE ordinal_score IS NOT NULL"""
        ):
            score_rows[row["paper_pair_id"]].append(row)
    finally:
        db.close()

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for paper_pair_id, paper in metadata.items():
        observations = score_rows.get(paper_pair_id, [])
        if not observations:
            continue
        key = pair_key(paper["orpha_id_a"], paper["orpha_id_b"])
        item = grouped.setdefault(
            key,
            {
                "orpha_id_a": key[0],
                "orpha_id_b": key[1],
                "disease_a": (
                    paper["disease_a"]
                    if key[0] == paper["orpha_id_a"]
                    else paper["disease_b"]
                ),
                "disease_b": (
                    paper["disease_b"]
                    if key[1] == paper["orpha_id_b"]
                    else paper["disease_a"]
                ),
                "paper_scores": [],
                "dimensions": defaultdict(list),
                "pmids": set(),
                "full_text_papers": set(),
            },
        )
        weight = sum(row["extraction_confidence"] for row in observations)
        paper_score = sum(
            (row["ordinal_score"] / 4.0) * row["extraction_confidence"]
            for row in observations
        ) / weight
        item["paper_scores"].append(paper_score)
        if paper["pmid"]:
            item["pmids"].add(paper["pmid"])
        if any(row["evidence_access_status"] == "full_text" for row in observations):
            item["full_text_papers"].add(paper["paper_id"])
        for row in observations:
            item["dimensions"][row["dimension"]].append(
                (row["ordinal_score"] / 4.0, row["extraction_confidence"])
            )

    result = []
    for item in grouped.values():
        record = {
            "orpha_id_a": item["orpha_id_a"],
            "orpha_id_b": item["orpha_id_b"],
            "disease_a": item["disease_a"],
            "disease_b": item["disease_b"],
            "literature_similarity": float(np.mean(item["paper_scores"])),
            "literature_min": float(np.min(item["paper_scores"])),
            "literature_max": float(np.max(item["paper_scores"])),
            "paper_observations": len(item["paper_scores"]),
            "unique_pmids": len(item["pmids"]),
            "full_text_papers": len(item["full_text_papers"]),
        }
        for dimension in DIMENSIONS:
            values = item["dimensions"].get(dimension, [])
            record[f"literature_{dimension}"] = (
                sum(score * weight for score, weight in values)
                / sum(weight for _, weight in values)
                if values
                else np.nan
            )
            record[f"observations_{dimension}"] = len(values)
        result.append(record)
    return sorted(result, key=lambda row: (row["orpha_id_a"], row["orpha_id_b"]))


def available_mean(
    similarities: np.ndarray,
    available: np.ndarray,
    modalities: tuple[str, ...],
    selected: tuple[str, ...],
) -> np.ndarray:
    columns = [modalities.index(name) for name in selected]
    numerator = (similarities[:, columns] * available[:, columns]).sum(axis=1)
    denominator = available[:, columns].sum(axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.full(len(similarities), np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def score_pairs(rows: list[dict[str, Any]], edges_path: Path) -> pd.DataFrame:
    model = load_model()
    model_ids = set(model.ids)
    evaluable = [
        row
        for row in rows
        if row["orpha_id_a"] in model_ids and row["orpha_id_b"] in model_ids
    ]
    a_ids = [row["orpha_id_a"] for row in evaluable]
    b_ids = [row["orpha_id_b"] for row in evaluable]
    similarities, available = model.engine.pair_features(a_ids, b_ids)
    logits = model.fusion.score(similarities[:, None, :], available[:, None, :])[:, 0]
    probabilities = model.fusion.probability(logits)
    percentiles = model.percentile(logits)

    edges = pd.read_csv(edges_path)
    edge_lookup = {
        pair_key(str(row.source), str(row.target)): row
        for row in edges.itertuples(index=False)
    }
    modalities = model.engine.modalities
    aligned = {
        dimension: available_mean(
            similarities, available, modalities, selected
        )
        for dimension, selected in ALIGNED_MODALITIES.items()
    }
    output = []
    for index, row in enumerate(evaluable):
        key = pair_key(row["orpha_id_a"], row["orpha_id_b"])
        edge = edge_lookup.get(key)
        known = model.known_relations(*key)
        result = dict(row)
        result.update(
            {
                "v2_logit": float(logits[index]),
                "v2_probability": float(probabilities[index]),
                "v2_percentile": float(percentiles[index]),
                "in_top10_graph": edge is not None,
                "graph_mutual": bool(edge.mutual) if edge is not None else False,
                "graph_support": str(edge.support) if edge is not None else "",
                "known_relations": "; ".join(sorted(known)),
            }
        )
        for column, modality in enumerate(modalities):
            result[f"v2_sim_{modality}"] = (
                float(similarities[index, column])
                if available[index, column]
                else np.nan
            )
        for dimension in DIMENSIONS:
            result[f"v2_aligned_{dimension}"] = float(aligned[dimension][index])
        output.append(result)
    return pd.DataFrame(output)


def strength_bin(score: float) -> str:
    if score <= 0.25:
        return "low_or_weak_0_to_0.25"
    if score <= 0.50:
        return "moderate_above_0.25_to_0.50"
    return "high_above_0.50"


def summarize(
    frame: pd.DataFrame,
    total_pairs: int,
    graph_summary: dict[str, Any],
) -> dict[str, Any]:
    frame = frame.copy()
    frame["literature_strength"] = frame["literature_similarity"].map(strength_bin)
    bins = []
    for name in (
        "low_or_weak_0_to_0.25",
        "moderate_above_0.25_to_0.50",
        "high_above_0.50",
    ):
        part = frame[frame["literature_strength"] == name]
        if part.empty:
            continue
        bins.append(
            {
                "bin": name,
                "pairs": int(len(part)),
                "median_v2_percentile": float(part["v2_percentile"].median()),
                "mean_v2_probability": float(part["v2_probability"].mean()),
                "top10_graph_pairs": int(part["in_top10_graph"].sum()),
                "top10_graph_rate": float(part["in_top10_graph"].mean()),
                "top_decile_rate": float((part["v2_percentile"] >= 0.90).mean()),
                "top_one_percent_rate": float(
                    (part["v2_percentile"] >= 0.99).mean()
                ),
            }
        )

    high_low = frame[
        (frame["literature_similarity"] > 0.50)
        | (frame["literature_similarity"] <= 0.25)
    ]
    labels = (high_low["literature_similarity"] > 0.50).astype(int)
    auc = (
        float(roc_auc_score(labels, high_low["v2_logit"]))
        if labels.nunique() == 2
        else None
    )
    dimensions = {}
    for dimension in DIMENSIONS:
        literature = frame[f"literature_{dimension}"]
        dimensions[dimension] = {
            "against_v2_overall_percentile": correlation(
                literature, frame["v2_percentile"]
            ),
            "against_aligned_v2_modalities": correlation(
                literature, frame[f"v2_aligned_{dimension}"]
            ),
            "aligned_modalities": list(ALIGNED_MODALITIES[dimension]),
        }

    possible_graph_pairs = (
        graph_summary["nodes"] * (graph_summary["nodes"] - 1) / 2
    )
    graph_edge_density = graph_summary["edges"] / possible_graph_pairs
    top10_rate = float(frame["in_top10_graph"].mean())
    top_decile_rate = float((frame["v2_percentile"] >= 0.90).mean())
    top_one_percent_rate = float((frame["v2_percentile"] >= 0.99).mean())
    return {
        "coverage": {
            "literature_pairs": total_pairs,
            "v2_evaluable_pairs": int(len(frame)),
            "unavailable_pairs": int(total_pairs - len(frame)),
            "fraction_evaluable": float(len(frame) / total_pairs)
            if total_pairs
            else 0.0,
        },
        "overall_agreement": {
            "literature_vs_v2_probability": correlation(
                frame["literature_similarity"], frame["v2_probability"]
            ),
            "literature_vs_v2_percentile": correlation(
                frame["literature_similarity"], frame["v2_percentile"]
            ),
            "high_vs_low_auc": auc,
            "high_definition": "literature_similarity > 0.50",
            "low_definition": "literature_similarity <= 0.25",
            "high_low_pairs": int(len(high_low)),
        },
        "graph_reflection": {
            "top10_graph_pairs": int(frame["in_top10_graph"].sum()),
            "top10_graph_rate": top10_rate,
            "all_graph_pair_edge_density": graph_edge_density,
            "top10_edge_enrichment_vs_random_pair": top10_rate
            / graph_edge_density,
            "mutual_top10_pairs": int(frame["graph_mutual"].sum()),
            "top_decile_pairs": int((frame["v2_percentile"] >= 0.90).sum()),
            "top_decile_rate": top_decile_rate,
            "top_decile_enrichment_vs_random_pair": top_decile_rate / 0.10,
            "top_one_percent_pairs": int((frame["v2_percentile"] >= 0.99).sum()),
            "top_one_percent_rate": top_one_percent_rate,
            "top_one_percent_enrichment_vs_random_pair": top_one_percent_rate
            / 0.01,
        },
        "by_literature_strength": bins,
        "by_dimension": dimensions,
        "limitations": [
            "The literature scores are automated and unverified.",
            "The aggregate target averages independently scored dimensions and is not a clinically validated overall similarity.",
            "The acquired corpus was selected by rare-disease relationship search terms, not random sampling.",
            "The top-10 graph is sparse; a missing edge does not mean v2 assigns zero similarity.",
            "Multiple papers may describe the same underlying study family.",
        ],
    }


def write_report(summary: dict[str, Any], path: Path) -> None:
    coverage = summary["coverage"]
    overall = summary["overall_agreement"]
    graph = summary["graph_reflection"]
    probability = overall["literature_vs_v2_probability"]
    percentile = overall["literature_vs_v2_percentile"]
    lines = [
        "# v2 graph versus acquired literature evidence",
        "",
        (
            f"Evaluated {coverage['v2_evaluable_pairs']:,} of "
            f"{coverage['literature_pairs']:,} unique literature pairs "
            f"({coverage['fraction_evaluable']:.1%})."
        ),
        "",
        "## Main result",
        "",
        (
            "The association is weak: literature score versus v2 probability "
            f"has Pearson {probability['pearson']:.3f} and Spearman "
            f"{probability['spearman']:.3f}; versus v2 random-pair percentile, "
            f"Pearson {percentile['pearson']:.3f} and Spearman "
            f"{percentile['spearman']:.3f}."
        ),
        (
            f"V2 distinguishes high (>0.50) from low/weak (<=0.25) literature "
            f"pairs with AUC {overall['high_vs_low_auc']:.3f}."
        ),
        "",
        "## Graph reflection",
        "",
        f"- Top-10 graph edges: {graph['top10_graph_pairs']:,} pairs ({graph['top10_graph_rate']:.1%}).",
        f"- Top-10 edge enrichment over a random catalogue pair: {graph['top10_edge_enrichment_vs_random_pair']:.1f}x.",
        f"- Mutual top-10 edges: {graph['mutual_top10_pairs']:,}.",
        f"- Top decile of random-pair scores: {graph['top_decile_pairs']:,} pairs ({graph['top_decile_rate']:.1%}).",
        f"- Top 1% of random-pair scores: {graph['top_one_percent_pairs']:,} pairs ({graph['top_one_percent_rate']:.1%}).",
        "",
        "## Interpretation limits",
        "",
        *[f"- {item}" for item in summary["limitations"]],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    database = args.database.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    literature = load_literature(database)
    frame = score_pairs(literature, V2_DATA_DIR / "graph/edges.csv")
    graph_summary = json.loads(
        (V2_DATA_DIR / "graph/graph_summary.json").read_text(encoding="utf-8")
    )
    summary = summarize(frame, len(literature), graph_summary)
    frame.to_csv(output_dir / "pair_results.csv", index=False)
    write_json(output_dir / "summary.json", summary)
    write_report(summary, output_dir / "REPORT.md")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
