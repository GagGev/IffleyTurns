"""Prospective evaluation of v3 against baselines.

Validation fold: knowledge before 2014-01-01, predict relations established
in 2014-2017 (used to choose the production stacker variant).
Test fold: knowledge before 2018-01-01, predict relations established from
2018 to the end of the data.  Nothing after a fold's cutoff is used to build
its inputs or train its models.

Usage:  python v3/run_evaluation.py [--folds validation test] [--seeds 3]
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import pandas as pd

from common import EVALUATION_DIR, configure_stdout, timed, write_json
from features import StaticSimilarity
from metrics import METRICS, PRIMARY_METRIC, bootstrap_mean, paired_difference
from world import describe, load_world

FOLDS = {
    "validation": ("2014-01-01", "2018-01-01"),
    "test": ("2018-01-01", None),
}
STATIC_MODELS = ("v1_weighted_jaccard", "v2_shipped", "v2_retrained", "gbm_static", "v3_neural", "v3_static")
REFERENCE_MODELS = ("random", "degree", "adamic_adar", "drug_mechanism")
STRATA = {
    "warm query": lambda d: d["query_warm"],
    "cold query": lambda d: ~d["query_warm"],
    "group query": lambda d: d["query_group"],
    "disease query": lambda d: ~d["query_group"],
    "oncology query": lambda d: d["query_oncology"],
    "non-oncology query": lambda d: ~d["query_oncology"],
}


def summarize_fold(per_query: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for gallery, frame in per_query.groupby("gallery"):
        models = {}
        for model, rows in frame.groupby("model", sort=False):
            models[model] = {"n_queries": int(len(rows))}
            for k, metric in enumerate(METRICS):
                mean, low, high = bootstrap_mean(rows[metric].to_numpy(), seed=k)
                models[model][metric] = {"mean": mean, "ci_low": low, "ci_high": high}
        out[gallery] = models
    return out


def paired_tests(per_query: pd.DataFrame, gallery: str, champion: str, others: list[str]) -> dict[str, Any]:
    frame = per_query[per_query["gallery"] == gallery]
    tests = {}
    for metric in (PRIMARY_METRIC, "auc"):
        wide = frame.pivot_table(index="query", columns="model", values=metric)
        if champion not in wide:
            continue
        for other in others:
            if other in wide and other != champion:
                tests.setdefault(other, {})[metric] = paired_difference(wide[champion].to_numpy(), wide[other].to_numpy())
    return tests


def strata_table(per_query: pd.DataFrame, gallery: str = "full") -> dict[str, Any]:
    frame = per_query[per_query["gallery"] == gallery]
    out = {}
    for stratum, select in STRATA.items():
        rows = frame[select(frame)]
        if rows.empty:
            continue
        out[stratum] = {
            "n_queries": int(rows["query"].nunique()),
            **{model: round(float(r[PRIMARY_METRIC].mean()), 4) for model, r in rows.groupby("model", sort=False)},
        }
    return out


def render_report(summary: dict[str, Any]) -> str:
    lines = ["# v3 prospective evaluation", ""]
    lines.append(
        "Ground truth: two rare diseases become related on the date the same drug holds FDA or EMA orphan "
        "designations for both (distinct regulatory decisions). Each fold trains only on knowledge before its "
        "cutoff and is scored on relations established afterwards. Metrics are per query disease, means with "
        "bootstrap 95% CIs; candidates exclude already-related and nested (group/member) diseases."
    )
    lines.append("")
    for fold, data in summary["folds"].items():
        info = data["info"]
        lines.append(f"## {fold} fold: cutoff {info['cutoff']}, until {info['until']}")
        lines.append("")
        lines.append(
            f"{info['known_relations']} relations known at the cutoff, {info['new_relations']} new relations to predict, "
            f"{info['queries_full']} query diseases, {info['warm_nodes']} diseases with a designation before the cutoff. "
            f"Stacker trained on {info['stacker_rows']} pairs ({info['stacker_positives']} positive) from origins "
            f"{', '.join(map(str, info['stacker_origins']))}."
        )
        lines.append("")
        for gallery, title in (
            ("full", "All 9,525 nodes as candidates"),
            ("warm", "Only diseases already designated before the cutoff as candidates"),
            ("full_approved", "All nodes; positives restricted to relations where both designations led to approval"),
        ):
            models = data["metrics"].get(gallery)
            if not models:
                continue
            lines.append(f"### {title} (`{gallery}`)")
            lines.append("")
            lines.append("| model | queries | MAP | AUC | MRR | Hits@10 | nDCG@10 | Recall@50 |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for model, m in models.items():
                lines.append(
                    f"| {model} | {m['n_queries']} | {m['map']['mean']:.3f} [{m['map']['ci_low']:.3f}, {m['map']['ci_high']:.3f}] "
                    f"| {m['auc']['mean']:.3f} | {m['mrr']['mean']:.3f} | {m['hits@10']['mean']:.3f} "
                    f"| {m['ndcg@10']['mean']:.3f} | {m['recall@50']['mean']:.3f} |"
                )
            lines.append("")
        tests = data.get("paired_tests", {})
        if tests:
            lines.append("### Paired comparisons (full gallery, per-query differences)")
            lines.append("")
            lines.append("| comparison | metric | mean difference [95% CI] | Wilcoxon p (greater) |")
            lines.append("|---|---|---|---|")
            for label, comparisons in tests.items():
                for other, metrics in comparisons.items():
                    for metric, t in metrics.items():
                        lines.append(
                            f"| {label} vs {other} | {metric} | {t['mean_difference']:+.4f} [{t['ci_low']:+.4f}, {t['ci_high']:+.4f}] "
                            f"| {t['p_wilcoxon_greater']:.2g} |"
                        )
            lines.append("")
        strata = data.get("strata", {})
        if strata:
            models = [m for m in next(iter(strata.values())) if m != "n_queries"]
            lines.append("### MAP by query stratum (full gallery)")
            lines.append("")
            lines.append("| stratum | queries | " + " | ".join(models) + " |")
            lines.append("|---|---|" + "---|" * len(models))
            for stratum, row in strata.items():
                lines.append(f"| {stratum} | {row['n_queries']} | " + " | ".join(f"{row.get(m, float('nan')):.3f}" for m in models) + " |")
            lines.append("")
        ranking = data.get("global_ranking", {})
        if ranking:
            base = ranking.get("base_rate", {})
            ks = sorted({k for m, v in ranking.items() if m != "base_rate" for k in v}, key=lambda s: int(s.split("@")[1]))
            lines.append(
                f"### Global ranking of new pairs among designated diseases (base rate {base.get('rate', 0):.4f}: "
                f"{base.get('positives', 0)} of {base.get('pairs', 0)} pairs)"
            )
            lines.append("")
            lines.append("| model | " + " | ".join(ks) + " |")
            lines.append("|---|" + "---|" * len(ks))
            for model, values in ranking.items():
                if model != "base_rate":
                    lines.append(f"| {model} | " + " | ".join(f"{values.get(k, float('nan')):.3f}" for k in ks) + " |")
            lines.append("")
        importance = data.get("stacker_importance", {})
        if importance:
            lines.append("### v3 stacker: permutation importance (drop in pooled AUC on training pairs)")
            lines.append("")
            lines.append("| feature family | AUC drop |")
            lines.append("|---|---|")
            for group, value in sorted(importance.items(), key=lambda x: -x[1]):
                lines.append(f"| {group} | {value:.4f} |")
            lines.append("")
    lines.append(f"Production stacker variant ({summary['selected']['selection']}): `{summary['selected']['stacker']}`")
    lines.append("")
    return "\n".join(lines)


def plot(summary: dict[str, Any], path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folds = [f for f in ("validation", "test") if f in summary["folds"]]
    fig, axes = plt.subplots(len(folds), 2, figsize=(14, 4.2 * len(folds)), squeeze=False)
    for r, fold in enumerate(folds):
        for c, gallery in enumerate(("full", "warm")):
            ax = axes[r][c]
            models = summary["folds"][fold]["metrics"].get(gallery, {})
            names = list(models)
            means = [models[m]["map"]["mean"] for m in names]
            low = [models[m]["map"]["mean"] - models[m]["map"]["ci_low"] for m in names]
            high = [models[m]["map"]["ci_high"] - models[m]["map"]["mean"] for m in names]
            colors = [
                "#c0392b" if m.startswith("v3") else "#e67e22" if m.startswith("stacker") else "#2980b9" if m in STATIC_MODELS else "#7f8c8d"
                for m in names
            ]
            ax.barh(names, means, xerr=[low, high], color=colors, capsize=2)
            ax.invert_yaxis()
            ax.set_xlabel("MAP (per query, 95% CI)")
            ax.set_title(f"{fold} fold, {gallery} gallery")
            ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def select_and_test(summary: dict[str, Any], per_query: pd.DataFrame) -> None:
    """Pick the production stacker on the first fold and test it against everything else."""

    from pipeline import STACKER_VARIANTS

    first = "validation" if "validation" in summary["folds"] else next(iter(summary["folds"]))
    validation = summary["folds"][first]["metrics"]["full"]
    best = max(STACKER_VARIANTS, key=lambda v: validation[v][PRIMARY_METRIC]["mean"] if v in validation else -1)
    summary["selected"] = {"stacker": best, "selection": f"highest full-gallery MAP on the {first} fold"}
    if best == "v3_stacker":
        return
    for fold, data in summary["folds"].items():
        frame = per_query[per_query["fold"] == fold]
        others = [m for m in (*REFERENCE_MODELS, *STATIC_MODELS, "v3_stacker") if m in set(frame["model"])]
        data["paired_tests"][best] = paired_tests(frame, "full", best, others)


def write_outputs(summary: dict[str, Any], per_query: pd.DataFrame) -> None:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    per_query.to_csv(EVALUATION_DIR / "per_query_metrics.csv", index=False)
    write_json(EVALUATION_DIR / "summary.json", summary)
    write_json(EVALUATION_DIR / "selected_config.json", summary["selected"])
    (EVALUATION_DIR / "report.md").write_text(render_report(summary), encoding="utf-8")
    plot(summary, EVALUATION_DIR / "map_by_model.png")


def main() -> int:
    configure_stdout()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folds", nargs="+", default=list(FOLDS), choices=list(FOLDS))
    parser.add_argument("--seeds", type=int, default=None, help="Static ensemble size (default from neural.DEFAULT_CONFIG)")
    parser.add_argument("--rebuild-world", action="store_true")
    parser.add_argument("--report-only", action="store_true", help="Re-render the report from saved per-query metrics.")
    args = parser.parse_args()

    if args.report_only:
        summary = json.loads((EVALUATION_DIR / "summary.json").read_text(encoding="utf-8"))
        per_query = pd.read_csv(EVALUATION_DIR / "per_query_metrics.csv")
        for data in summary["folds"].values():
            data["paired_tests"] = {k: v for k, v in data["paired_tests"].items() if k in ("v3_stacker", "v3_static")}
        select_and_test(summary, per_query)
        write_outputs(summary, per_query)
        print(f"Report: {EVALUATION_DIR / 'report.md'}")
        return 0

    from pipeline import evaluate_fold

    config = {"seeds": args.seeds} if args.seeds else None
    world = load_world(rebuild=args.rebuild_world)
    sim = StaticSimilarity.from_world(world)
    end_of_data = pd.Timestamp(world.regulatory.designations["date"].max())
    summary: dict[str, Any] = {"world": describe(world), "end_of_data": str(end_of_data.date()), "folds": {}}
    frames = []
    for fold in args.folds:
        start, until = FOLDS[fold]
        with timed(f"{fold} fold"):
            result = evaluate_fold(
                world,
                sim,
                fold,
                pd.Timestamp(start),
                pd.Timestamp(until) if until else None,
                end_of_data,
                config,
            )
        frames.append(result.per_query)
        others = [m for m in result.per_query["model"].unique() if m != "v3_stacker"]
        summary["folds"][fold] = {
            "info": result.info,
            "metrics": summarize_fold(result.per_query),
            "paired_tests": {
                "v3_stacker": paired_tests(result.per_query, "full", "v3_stacker", others),
                "v3_static": paired_tests(result.per_query, "full", "v3_static", [m for m in STATIC_MODELS if m != "v3_static"]),
            },
            "strata": strata_table(result.per_query),
            "global_ranking": result.global_ranking,
            "stacker_importance": result.importance,
        }
    per_query = pd.concat(frames)
    select_and_test(summary, per_query)
    write_outputs(summary, per_query)
    for fold, data in summary["folds"].items():
        print(f"\n{fold}: full-gallery MAP / AUC")
        for model, m in data["metrics"]["full"].items():
            print(f"  {model:22s} {m['map']['mean']:.4f}  {m['auc']['mean']:.4f}  (n={m['n_queries']})")
    print(f"\nReport: {EVALUATION_DIR / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
