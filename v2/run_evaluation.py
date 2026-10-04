"""Train and evaluate v2 disease-similarity models on database-derived benchmarks.

Protocol
--------
1. Diseases are split 60/20/20 into train/validation/test by a stable hash.
2. Encoders (IDF / information content, text IDF) and fusion models are fitted
   on training diseases only.
3. For each benchmark, each validation/test disease is a query: all other
   eligible diseases are ranked, and curated relations of the query are the
   positives.  Modalities derived from the benchmark's label source are masked.
4. Hyperparameters are chosen on validation queries (macro MAP over
   benchmarks); test queries are evaluated once.
5. A "newly described disease" stress test additionally hides what a new
   disease usually lacks (Open Targets associations, trial drugs and their
   targets, a place in the Orphanet/Mondo classification).

Run:  python v2/run_evaluation.py
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from benchmarks import BENCHMARK_SPECS, OrphanetIndex, Relation, build_relations, controls_mask
from common import EVALUATION_DIR, configure_stdout, timed, write_json
from data_sources import Bundle, load_bundle
from metrics import METRICS, PRIMARY_METRIC, paired_difference, pooled_auc, query_metrics, summarize
from modalities import MODALITIES, MODALITY_DESCRIPTIONS, DiseaseEncoder
from scoring import (
    PatternFusion,
    RandomModel,
    SimilarityEngine,
    SingleModality,
    UniformFusion,
    V1Baseline,
    build_training_set,
    fit_gbm,
    fit_logistic,
    subset,
)

CHUNK = 128
RANDOM_REPEATS = 10
CONTROLS_PER_QUERY = 50
TEXT_MODALITIES = ("name", "text")
NEW_DISEASE_HIDDEN = ("ot_gene", "drug", "drug_target", "ontology")
LOGISTIC_GRID = [(C, avail) for C in (0.03, 0.3, 3.0) for avail in (True, False)]
SHRINKAGE_GRID = (0.0, 0.5, 1.0, 2.0)
MAIN_MODELS = ("random", "v1_weighted_jaccard", "uniform_mean", "logistic_single_task", "gbm_fusion", "logistic_fusion")
REPORT_MODELS = (*MAIN_MODELS[:-1], "logistic_unconstrained", MAIN_MODELS[-1], "pattern_fusion")
NEW_DISEASE_MODELS = ("uniform_mean", "gbm_fusion", "logistic_fusion", "pattern_fusion")
MODEL_LABELS = {
    "random": "Random ranking",
    "v1_weighted_jaccard": "v1 weighted Jaccard (previous approach)",
    "uniform_mean": "v2 uniform mean of modality similarities",
    "logistic_single_task": "v2 logistic fusion, trained on this benchmark only",
    "gbm_fusion": "v2 gradient-boosted fusion (multi-task)",
    "logistic_unconstrained": "v2 logistic fusion (multi-task) without non-negative weights",
    "logistic_fusion": "v2 logistic fusion (multi-task, non-negative weights) -- production graph model",
    "pattern_fusion": "v2 logistic fusion refitted to the query's available modalities",
}


@dataclass
class Context:
    bundle: Bundle
    engine: SimilarityEngine
    v1: V1Baseline
    orphanet: OrphanetIndex
    split: dict[str, str]


def evaluate(
    ctx: Context,
    relation: Relation,
    queries: Sequence[str],
    models: dict[str, Any],
    variants: Optional[dict[str, tuple[Sequence[str], dict[str, Any]]]] = None,
    with_controls: bool = False,
    with_hard: bool = False,
    with_baselines: bool = True,
    seed: int = 0,
) -> dict[str, Any]:
    """Rank the benchmark gallery for every query with every model.

    ``variants`` maps a suffix to (extra hidden modalities, models): those
    models are also scored with the extra modalities hidden, as ``name_suffix``.
    """

    spec = relation.spec
    variants = variants or {}
    gallery = ctx.engine.gallery(sorted(relation.eligible))
    gallery_ids = np.array(gallery.ids)
    baselines = ["random", "v1_weighted_jaccard"] if with_baselines else []
    names = [*baselines, *models, *[f"{n}_{suffix}" for suffix, (_, extra) in variants.items() for n in extra]]
    rows: dict[str, list[np.ndarray]] = {n: [] for n in names}
    hard_rows: dict[str, list[np.ndarray]] = {n: [] for n in names}
    hard_queries: list[str] = []
    control_scores: dict[str, tuple[list[float], list[float]]] = {n: ([], []) for n in names}
    random_model = RandomModel(seed)
    control_rng = np.random.default_rng(seed + 7)

    for start in range(0, len(queries), CHUNK):
        chunk = list(queries[start : start + CHUNK])
        Qm, Qa = ctx.engine.rows(chunk)
        S, A = ctx.engine.block(Qm, Qa, gallery, masked=spec.masked)
        scores = {name: model.score(S, A) for name, model in models.items()}
        for suffix, (hidden, extra) in variants.items():
            columns = [ctx.engine.modalities.index(m) for m in hidden]
            S2, A2 = S.copy(), A.copy()
            S2[:, :, columns] = 0.0
            A2[:, :, columns] = False
            for name, model in extra.items():
                scores[f"{name}_{suffix}"] = model.score(S2, A2)
        random_scores = []
        if with_baselines:
            scores["v1_weighted_jaccard"] = ctx.v1.block(chunk, gallery.ids, spec.v1_masked)
            random_scores = [random_model.score(S, A) for _ in range(RANDOM_REPEATS)]

        for i, query in enumerate(chunk):
            valid = gallery_ids != query
            positives = np.isin(gallery_ids, list(relation.neighbors[query]))
            strata = [("all", valid, positives, rows)]
            hard = relation.hard_neighbors.get(query, set()) if with_hard else set()
            if hard:
                hard_mask = np.isin(gallery_ids, list(hard))
                strata.append(("hard", valid & ~(positives & ~hard_mask), hard_mask, hard_rows))
                hard_queries.append(query)
            query_seed = seed * 1_000_003 + start + i
            for _, candidates, labels, store in strata:
                y = labels[candidates]
                for name in names:
                    if name == "random":
                        values = np.mean(
                            [
                                query_metrics(r[i][candidates], y, np.random.default_rng(query_seed + 17 * k))
                                for k, r in enumerate(random_scores)
                            ],
                            axis=0,
                        )
                    else:
                        values = query_metrics(scores[name][i][candidates], y, np.random.default_rng(query_seed))
                    store[name].append(values)
            if with_controls:
                controls = np.flatnonzero(controls_mask(ctx.bundle, ctx.orphanet, query, gallery.ids))
                if controls.size > CONTROLS_PER_QUERY:
                    controls = control_rng.choice(controls, CONTROLS_PER_QUERY, replace=False)
                positive_idx = np.flatnonzero(positives & valid)
                for name in names:
                    if name == "random":
                        values = random_scores[0][i]
                    else:
                        values = scores[name][i]
                    control_scores[name][0].extend(values[positive_idx].tolist())
                    control_scores[name][1].extend(values[controls].tolist())

    result = {
        "queries": list(queries),
        "all": {n: np.vstack(v) for n, v in rows.items()},
        "gallery_size": len(gallery_ids),
    }
    if with_hard and hard_queries:
        result["hard"] = {n: np.vstack(v) for n, v in hard_rows.items()}
        result["hard_queries"] = hard_queries
    if with_controls:
        result["controls"] = {
            n: {
                "pair_auc": pooled_auc(*control_scores[n]),
                "n_related_pairs": len(control_scores[n][0]),
                "n_control_pairs": len(control_scores[n][1]),
            }
            for n in names
        }
    return result


def split_ids(ctx: Context, name: str) -> list[str]:
    return sorted(d for d, s in ctx.split.items() if s == name)


def fmt(summary: dict[str, float], digits: int = 3) -> str:
    return f"{summary['mean']:.{digits}f} [{summary['ci_low']:.{digits}f}, {summary['ci_high']:.{digits}f}]"


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_stdout()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--negatives", type=int, default=10, help="Sampled negative pairs per positive pair.")
    parser.add_argument("--max-queries", type=int, default=0, help="Subsample queries per benchmark (0 = all).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    with timed("Loading database-derived disease records"):
        bundle = load_bundle()
    ids = sorted(bundle.records)
    split = bundle.meta["split"].to_dict()
    train_ids = {d for d in ids if split[d] == "train"}
    with timed("Fitting modality encoders on training diseases"):
        encoder = DiseaseEncoder(bundle.knowledge).fit([bundle.records[d] for d in sorted(train_ids)], [bundle.records[d] for d in ids])
        engine = SimilarityEngine(encoder.transform([bundle.records[d] for d in ids]), ids)
    ctx = Context(bundle, engine, V1Baseline(bundle.v1_sets, ids), OrphanetIndex(bundle), split)
    relations = build_relations(bundle)
    rel_list = list(relations.values())
    rng = np.random.default_rng(args.seed)

    def queries_for(relation: Relation, split_name: str) -> list[str]:
        queries = relation.queries(split_ids(ctx, split_name))
        if args.max_queries and len(queries) > args.max_queries:
            queries = sorted(rng.choice(queries, args.max_queries, replace=False).tolist())
        return queries

    with timed("Building masked multi-task training pairs (train diseases only)"):
        train = build_training_set(engine, rel_list, train_ids, args.negatives, args.seed)
    with timed("Fitting fusion models"):
        logistic_candidates = {
            f"logistic_C{C}_{'avail' if avail else 'noavail'}": fit_logistic(train, engine.modalities, C, avail)
            for C, avail in LOGISTIC_GRID
        }
        gbm = fit_gbm(train, seed=args.seed)
        uniform_candidates = {f"uniform_shrink{s}": UniformFusion(s) for s in SHRINKAGE_GRID}

    selection: dict[str, dict[str, float]] = defaultdict(dict)
    with timed("Model selection on validation diseases"):
        for relation in rel_list:
            queries = queries_for(relation, "validation")
            result = evaluate(ctx, relation, queries, {**logistic_candidates, **uniform_candidates, "gbm_fusion": gbm}, seed=args.seed)
            column = METRICS.index(PRIMARY_METRIC)
            for name, values in result["all"].items():
                selection[name][relation.name] = float(values[:, column].mean())
            print(f"    {relation.name}: {len(queries)} validation queries")
    macro = {name: float(np.mean(list(v.values()))) for name, v in selection.items()}
    best_logistic = max(logistic_candidates, key=lambda n: macro[n])
    best_uniform = max(uniform_candidates, key=lambda n: macro[n])
    production = logistic_candidates[best_logistic]
    production.name = "logistic_fusion"
    print(f"[v2] selected {best_logistic} and {best_uniform} (validation macro {PRIMARY_METRIC})")
    unconstrained = fit_logistic(
        train, engine.modalities, production.params["C"], production.params["use_availability"],
        nonnegative=False, name="logistic_unconstrained",
    )
    pattern = PatternFusion(train, production)

    pattern_check: dict[str, dict[str, float]] = defaultdict(dict)
    with timed("Validation: refitted-to-available-modalities fusion versus the general fusion"):
        candidates = {"logistic_fusion": production, "pattern_fusion": pattern}
        for relation in rel_list:
            result = evaluate(
                ctx, relation, queries_for(relation, "validation"), candidates,
                variants={"new_disease": (NEW_DISEASE_HIDDEN, candidates)}, with_baselines=False, seed=args.seed,
            )
            column = METRICS.index(PRIMARY_METRIC)
            for name, values in result["all"].items():
                pattern_check[name][relation.name] = float(values[:, column].mean())
    pattern_macro = {name: float(np.mean(list(v.values()))) for name, v in pattern_check.items()}
    new_disease_fusion = (
        "pattern" if pattern_macro["pattern_fusion_new_disease"] >= pattern_macro["logistic_fusion_new_disease"] else "general"
    )
    print(f"[v2] new-disease placement will use the {new_disease_fusion} fusion (validation macro {PRIMARY_METRIC}: "
          f"pattern {pattern_macro['pattern_fusion_new_disease']:.3f} vs general {pattern_macro['logistic_fusion_new_disease']:.3f})")

    test_results: dict[str, Any] = {}
    with timed("Final evaluation on test diseases"):
        for b, relation in enumerate(rel_list):
            single_task = fit_logistic(
                subset(train, b), engine.modalities, production.params["C"], production.params["use_availability"]
            )
            allowed = [m for m in engine.modalities if m not in relation.spec.masked]
            models = {
                "uniform_mean": uniform_candidates[best_uniform],
                "logistic_single_task": single_task,
                "gbm_fusion": gbm,
                "logistic_unconstrained": unconstrained,
                "logistic_fusion": production,
                "pattern_fusion": pattern,
                **{f"only_{m}": SingleModality(m, engine.modalities) for m in allowed},
            }
            variants = {
                "no_text": (TEXT_MODALITIES, {n: models[n] for n in ("uniform_mean", "gbm_fusion", "logistic_fusion")}),
                "new_disease": (NEW_DISEASE_HIDDEN, {n: models[n] for n in NEW_DISEASE_MODELS}),
            }
            queries = queries_for(relation, "test")
            test_results[relation.name] = evaluate(
                ctx,
                relation,
                queries,
                models,
                variants=variants,
                with_controls=True,
                with_hard=relation.name != "orphanet_siblings",
                seed=args.seed,
            )
            print(f"    {relation.name}: {len(queries)} test queries")
    print(f"[v2] fitted {len(pattern.submodels) - 1} pattern submodels")

    write_outputs(
        args, bundle, engine, relations, selection, macro, best_logistic, best_uniform, production, unconstrained,
        pattern, {"per_benchmark": pattern_check, "macro": pattern_macro, "new_disease_fusion": new_disease_fusion},
        test_results,
    )
    return 0


def write_outputs(
    args, bundle, engine, relations, selection, macro, best_logistic, best_uniform, production, unconstrained,
    pattern, pattern_check, test_results,
) -> None:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    flat_rows = []
    per_query_rows = []
    for bench, result in test_results.items():
        summaries[bench] = {}
        comparisons[bench] = {}
        for stratum in ("all", "hard"):
            if stratum not in result:
                continue
            block = result[stratum]
            queries = result["queries"] if stratum == "all" else result["hard_queries"]
            summaries[bench][stratum] = {
                "n_queries": len(queries),
                "mean_positives_per_query": float(
                    np.mean(
                        [
                            len(relations[bench].neighbors[q] if stratum == "all" else relations[bench].hard_neighbors[q])
                            for q in queries
                        ]
                    )
                ),
                "models": {},
            }
            comparisons[bench][stratum] = {}
            for name, values in block.items():
                summary = summarize(values, seed=args.seed)
                summaries[bench][stratum]["models"][name] = summary
                for metric, s in summary.items():
                    flat_rows.append(
                        {"benchmark": bench, "stratum": stratum, "model": name, "metric": metric, **s, "n_queries": len(queries)}
                    )
                if name in ("random", "v1_weighted_jaccard"):
                    continue
                comparisons[bench][stratum][name] = {
                    reference: {
                        metric: paired_difference(values[:, j], block[reference][:, j], seed=args.seed + j)
                        for j, metric in enumerate(METRICS)
                        if metric in ("auc", "map", "mrr", "ndcg@10")
                    }
                    for reference in ("random", "v1_weighted_jaccard")
                }
            for suffix in ("", "_new_disease"):
                comparisons[bench][stratum][f"pattern_vs_general{suffix}"] = {
                    metric: paired_difference(
                        block[f"pattern_fusion{suffix}"][:, j], block[f"logistic_fusion{suffix}"][:, j], seed=args.seed + j
                    )
                    for j, metric in enumerate(METRICS)
                    if metric in ("auc", "map")
                }
            if stratum == "all":
                for name in REPORT_MODELS:
                    for q, values in zip(queries, block[name]):
                        per_query_rows.append({"benchmark": bench, "query": q, "model": name, **dict(zip(METRICS, values))})

    controls = {bench: result.get("controls", {}) for bench, result in test_results.items()}
    data_summary = {
        "cohort_size": len(bundle.records),
        "split_sizes": bundle.meta["split"].value_counts().to_dict(),
        "modality_coverage": {m: int(engine.available[:, j].sum()) for j, m in enumerate(engine.modalities)},
        "benchmarks": {
            name: {
                "title": r.spec.title,
                "description": r.spec.description,
                "masked_modalities": list(r.spec.masked),
                "v1_masked_families": list(r.spec.v1_masked),
                "eligible_diseases": len(r.eligible),
                "related_pairs": r.n_pairs(),
                "pairs_outside_orphanet_group": sum(len(v) for v in r.hard_neighbors.values()) // 2 if r.hard_neighbors else None,
                "gallery_size": test_results[name]["gallery_size"],
            }
            for name, r in relations.items()
        },
        "training_pairs": {
            name: {
                "positives": int(((pattern.data.benchmark == b) & (pattern.data.y == 1)).sum()),
                "negatives": int(((pattern.data.benchmark == b) & (pattern.data.y == 0)).sum()),
            }
            for b, name in enumerate(relations)
        },
    }
    selected = {
        "logistic": best_logistic,
        "logistic_params": production.params,
        "uniform": best_uniform,
        "new_disease_fusion": pattern_check["new_disease_fusion"],
        "negatives_per_positive": args.negatives,
        "seed": args.seed,
    }
    document = {
        "protocol": {
            "split": "disease-level stable hash, 60/20/20 train/validation/test",
            "inductive": "encoders and fusion models fitted on training diseases only; every test query is an unseen disease",
            "selection": f"validation macro-average {PRIMARY_METRIC} over benchmarks",
            "metrics": list(METRICS),
            "uncertainty": "bootstrap over queries (2000 resamples), paired for differences; one-sided Wilcoxon p-values",
            "max_queries": args.max_queries,
        },
        "data": data_summary,
        "selected": selected,
        "validation_selection": {"per_benchmark": selection, "macro": macro},
        "validation_pattern_check": pattern_check,
        "new_disease_hidden_modalities": list(NEW_DISEASE_HIDDEN),
        "production_model_coefficients_train_split": production.coefficients(),
        "unconstrained_model_coefficients_train_split": unconstrained.coefficients(),
        "new_disease_submodel_coefficients_train_split": pattern.submodel(NEW_DISEASE_HIDDEN).coefficients(),
        "test": summaries,
        "comparisons": comparisons,
        "controls_pair_auc": controls,
    }
    write_json(EVALUATION_DIR / "metrics.json", document)
    write_json(EVALUATION_DIR / "selected_config.json", selected)
    with (EVALUATION_DIR / "test_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    with (EVALUATION_DIR / "per_query_test.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_query_rows[0]))
        writer.writeheader()
        writer.writerows(per_query_rows)
    plot_results(summaries)
    write_report(document)
    print(f"[v2] wrote evaluation outputs to {EVALUATION_DIR}")


def plot_results(summaries: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(MAIN_MODELS)
    colors = ["#9e9e9e", "#d98c5f", "#8fb3d9", "#7fbf7f", "#b39ddb", "#2e6fba"]
    fig, axes = plt.subplots(2, len(summaries), figsize=(5.2 * len(summaries), 7.5), squeeze=False)
    for col, (bench, by_stratum) in enumerate(summaries.items()):
        stats = by_stratum["all"]["models"]
        for row, metric in enumerate(("auc", "map")):
            ax = axes[row][col]
            means = [stats[m][metric]["mean"] for m in models]
            low = [stats[m][metric]["mean"] - stats[m][metric]["ci_low"] for m in models]
            high = [stats[m][metric]["ci_high"] - stats[m][metric]["mean"] for m in models]
            ax.bar(range(len(models)), means, yerr=[low, high], color=colors, capsize=3)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(["random", "v1", "uniform", "logit\n1-task", "GBM", "logit\nmulti"], fontsize=8)
            ax.set_ylabel({"auc": "per-query AUROC", "map": "mean average precision"}[metric])
            if metric == "auc":
                ax.axhline(0.5, color="black", lw=0.8, ls="--")
                ax.set_ylim(0.4, 1.0)
            ax.set_title(f"{BENCHMARK_SPECS[bench].title}\n(n={by_stratum['all']['n_queries']} test diseases)", fontsize=10)
    fig.suptitle("v2 retrieval of curated relations for unseen test diseases (95% bootstrap CI)", fontsize=12)
    fig.tight_layout()
    fig.savefig(EVALUATION_DIR / "benchmark_results.png", dpi=130)
    plt.close(fig)


def write_report(doc: dict[str, Any]) -> None:
    lines: list[str] = []
    add = lines.append
    data = doc["data"]
    add("# v2 evaluation report\n")
    add("Generated by `python v2/run_evaluation.py`. All numbers are on **test diseases that no encoder or model saw "
        "during fitting** (the new-disease setting), with 95% bootstrap confidence intervals over query diseases.\n")
    add("## Headline\n")
    add("| Benchmark | Test queries | Random MAP | v1 MAP | v2 production MAP | v2 production AUROC | Δ MAP vs random [95% CI] | Δ MAP vs v1 [95% CI] |")
    add("|---|---|---|---|---|---|---|---|")
    for bench, s in doc["test"].items():
        m = s["all"]["models"]
        c = doc["comparisons"][bench]["all"]["logistic_fusion"]
        add(
            f"| {BENCHMARK_SPECS[bench].title} | {s['all']['n_queries']} | {m['random']['map']['mean']:.3f} | "
            f"{m['v1_weighted_jaccard']['map']['mean']:.3f} | {fmt(m['logistic_fusion']['map'])} | {fmt(m['logistic_fusion']['auc'])} | "
            f"{c['random']['map']['mean_difference']:+.3f} [{c['random']['map']['ci_low']:+.3f}, {c['random']['map']['ci_high']:+.3f}] | "
            f"{c['v1_weighted_jaccard']['map']['mean_difference']:+.3f} [{c['v1_weighted_jaccard']['map']['ci_low']:+.3f}, {c['v1_weighted_jaccard']['map']['ci_high']:+.3f}] |"
        )
    add("\n![benchmark results](benchmark_results.png)\n")

    add("## Evaluation design\n")
    add(f"- Cohort: {data['cohort_size']:,} active Orphanet disorders and subtypes (groups of disorders, obsolete entries, "
        "and diseases flagged non-rare in Europe excluded).")
    add(f"- Disease-level split: {data['split_sizes']}. Encoders (IDF / information content) and fusion models are fitted on "
        "training diseases only.")
    add("- Each test disease is a query; all other eligible diseases are ranked; curated relations are positives. "
        "Modalities derived from a benchmark's own label source are **masked** so the relation must be recovered from "
        "independent evidence.")
    add(f"- Hyperparameters chosen on validation diseases ({doc['protocol']['selection']}): {doc['selected']}.\n")
    add("| Benchmark | Label source | Eligible diseases | Related pairs | Pairs outside a shared Orphanet group | Masked v2 modalities | Masked v1 families |")
    add("|---|---|---|---|---|---|---|")
    for bench, info in data["benchmarks"].items():
        add(
            f"| {info['title']} | {info['description']} | {info['eligible_diseases']:,} | {info['related_pairs']:,} | "
            f"{info['pairs_outside_orphanet_group'] if info['pairs_outside_orphanet_group'] is not None else 'n/a'} | "
            f"{', '.join(info['masked_modalities'])} | {', '.join(info['v1_masked_families'])} |"
        )
    add("\nModalities:\n")
    for m in MODALITIES:
        add(f"- `{m}` ({data['modality_coverage'].get(m, 0):,} diseases): {MODALITY_DESCRIPTIONS[m]}")
    add("")

    add("## Results by benchmark (test diseases)\n")
    add("AUROC is per query (0.5 = chance). MAP = mean average precision, MRR = mean reciprocal rank of the first related disease, "
        "Hits@10 = share of queries with a related disease in the top 10.\n")
    for bench, s in doc["test"].items():
        add(f"### {BENCHMARK_SPECS[bench].title}\n")
        add(f"{s['all']['n_queries']} test queries, {s['all']['mean_positives_per_query']:.1f} related diseases per query on average, "
            f"gallery of {data['benchmarks'][bench]['gallery_size']:,} diseases.\n")
        add("| Model | AUROC | MAP | MRR | Hits@10 | nDCG@10 |")
        add("|---|---|---|---|---|---|")
        models = s["all"]["models"]
        for name in REPORT_MODELS:
            m = models[name]
            add(f"| {MODEL_LABELS[name]} | {fmt(m['auc'])} | {fmt(m['map'])} | {fmt(m['mrr'])} | {m['hits@10']['mean']:.3f} | {m['ndcg@10']['mean']:.3f} |")
        add("\nSingle-modality ablation (unmasked modalities only):\n")
        add("| Modality | AUROC | MAP |")
        add("|---|---|---|")
        singles = sorted((n for n in models if n.startswith("only_")), key=lambda n: -models[n]["map"]["mean"])
        for name in singles:
            add(f"| {name[5:]} | {fmt(models[name]['auc'])} | {fmt(models[name]['map'])} |")
        add("\nWithout free text (name and description masked too):\n")
        add("| Model | AUROC | MAP |")
        add("|---|---|---|")
        for name in ("uniform_mean", "gbm_fusion", "logistic_fusion"):
            m = models[f"{name}_no_text"]
            add(f"| {MODEL_LABELS[name]} | {fmt(m['auc'])} | {fmt(m['map'])} |")
        if "hard" in s:
            h = s["hard"]
            add(f"\nHarder relations -- related diseases that do **not** share an Orphanet parent group or lineage with the query "
                f"({h['n_queries']} queries, {h['mean_positives_per_query']:.1f} positives each; Orphanet-close positives removed from the ranking):\n")
            add("| Model | AUROC | MAP | MRR |")
            add("|---|---|---|---|")
            for name in REPORT_MODELS:
                m = h["models"][name]
                add(f"| {MODEL_LABELS[name]} | {fmt(m['auc'])} | {fmt(m['map'])} | {fmt(m['mrr'])} |")
        add("")

    add("## Known relations versus clearly unrelated controls\n")
    add("PROJECT.md's initial evaluation asks whether known related pairs score above clearly unrelated controls. Controls pair "
        "each test query with diseases from a different top-level Orphanet category that share no Orphanet group, curated gene, "
        "or drug with it. The value is the probability that a related pair outranks a control pair (pair-level AUROC).\n")
    add("| Model | " + " | ".join(BENCHMARK_SPECS[b].title for b in doc["controls_pair_auc"]) + " |")
    add("|---|" + "---|" * len(doc["controls_pair_auc"]))
    for name in REPORT_MODELS:
        add(f"| {MODEL_LABELS[name]} | " + " | ".join(f"{doc['controls_pair_auc'][b][name]['pair_auc']:.3f}" for b in doc["controls_pair_auc"]) + " |")
    add("")

    hidden = ", ".join(doc["new_disease_hidden_modalities"])
    check = doc["validation_pattern_check"]
    add("## Placing newly described diseases\n")
    add(f"A newly described disease usually has phenotypes, genes, a description, inheritance and onset, but no Open Targets "
        f"association profile, no trial drugs and no place in the Orphanet/Mondo classification. In this stress test the test "
        f"queries are scored with those modalities ({hidden}) hidden as well as the benchmark's own. The refitted fusion "
        "re-learns the weights, on the same training pairs, with each query's missing modalities hidden, so their weight moves "
        "to correlated modalities the query does have (for example from Open Targets genes and pathways to the curated gene).\n")
    add(f"On validation diseases this setting gave macro MAP {check['macro']['pattern_fusion_new_disease']:.3f} for the refitted "
        f"fusion versus {check['macro']['logistic_fusion_new_disease']:.3f} for the general fusion, so `place_disease.py` uses the "
        f"**{check['new_disease_fusion']}** fusion.\n")
    short = {"uniform_mean": "Uniform mean", "gbm_fusion": "Gradient boosting", "logistic_fusion": "General logistic (graph)",
             "pattern_fusion": "Refitted logistic"}
    add("| Benchmark | Random | " + " | ".join(short[n] for n in NEW_DISEASE_MODELS) + " | Δ MAP refitted vs general [95% CI] |")
    add("|---|---|" + "---|" * len(NEW_DISEASE_MODELS) + "---|")
    for bench, s in doc["test"].items():
        m = s["all"]["models"]
        c = doc["comparisons"][bench]["all"]["pattern_vs_general_new_disease"]["map"]
        cells = " | ".join(f"MAP {m[f'{n}_new_disease']['map']['mean']:.3f}, AUROC {m[f'{n}_new_disease']['auc']['mean']:.3f}" for n in NEW_DISEASE_MODELS)
        add(f"| {BENCHMARK_SPECS[bench].title} | MAP {m['random']['map']['mean']:.3f} | {cells} | "
            f"{c['mean_difference']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] |")
    add("\nWith full catalogue annotations (the standard setting above), refitted minus general fusion:\n")
    add("| Benchmark | Δ AUROC [95% CI] | Δ MAP [95% CI] |")
    add("|---|---|---|")
    for bench in doc["test"]:
        c = doc["comparisons"][bench]["all"]["pattern_vs_general"]
        add(f"| {BENCHMARK_SPECS[bench].title} | {c['auc']['mean_difference']:+.3f} [{c['auc']['ci_low']:+.3f}, {c['auc']['ci_high']:+.3f}] | "
            f"{c['map']['mean_difference']:+.3f} [{c['map']['ci_low']:+.3f}, {c['map']['ci_high']:+.3f}] |")
    add("")

    add("## Learned modality weights (production logistic fusion, fitted on training diseases)\n")
    add("Each weight is learned only from benchmarks whose labels do not come from that modality's source. Similarity weights "
        "are constrained to be non-negative, so more overlap in any modality never lowers a score; without the constraint, "
        "correlated modalities (gene, pathway, Open Targets genes) receive offsetting positive and negative weights that "
        "would make edge explanations misleading. Availability weights shift the score according to which modalities are "
        "annotated for both diseases, independently of overlap. The last column is the refitted fusion for a query without "
        f"{hidden}.\n")
    add("| Modality | Similarity weight | Availability weight | Unconstrained similarity weight | Similarity weight, new-disease profile |")
    add("|---|---|---|---|---|")
    unconstrained = {r["modality"]: r for r in doc["unconstrained_model_coefficients_train_split"]}
    profile = {r["modality"]: r for r in doc["new_disease_submodel_coefficients_train_split"]}
    for row in sorted(doc["production_model_coefficients_train_split"], key=lambda r: -r["similarity_weight"]):
        refit = "hidden" if row["modality"] in doc["new_disease_hidden_modalities"] else f"{profile[row['modality']]['similarity_weight']:+.2f}"
        add(
            f"| {row['modality']} | {row['similarity_weight']:+.2f} | {row['availability_weight']:+.2f} | "
            f"{unconstrained[row['modality']]['similarity_weight']:+.2f} | {refit} |"
        )
    add("")
    add("## Statistical comparison (paired, test diseases)\n")
    add("| Benchmark | Model | Δ AUROC vs random | p | Δ MAP vs v1 | p |")
    add("|---|---|---|---|---|---|")
    for bench, by_stratum in doc["comparisons"].items():
        for name in ("uniform_mean", "gbm_fusion", "logistic_unconstrained", "logistic_fusion"):
            c = by_stratum["all"][name]
            add(
                f"| {BENCHMARK_SPECS[bench].title} | {name} | {c['random']['auc']['mean_difference']:+.3f} | {c['random']['auc']['p_wilcoxon_greater']:.1e} | "
                f"{c['v1_weighted_jaccard']['map']['mean_difference']:+.3f} | {c['v1_weighted_jaccard']['map']['p_wilcoxon_greater']:.1e} |"
            )
    add("")
    add("## Caveats\n")
    add("- Benchmarks are curated database relations, not clinical ground truth; unlabeled pairs are treated as unrelated, so "
        "precision is a lower bound (a top-ranked 'false positive' may be a real, unrecorded relation).")
    add("- Masking removes the label's own source, but sources are not fully independent (for example, Orphanet experts may "
        "group allelic disorders together, so the classification modality helps predict shared genes).")
    add("- Orphanet descriptions and names can mention mechanisms; gene symbols, cytobands, and the disease's own names are "
        "stripped, and the 'without free text' tables show results with both text modalities removed.")
    add("- The literature review data in `literature_review/` (and the literature-derived `.data/literature_acquisition/`, "
        "`.data/splits/`, `.data/models/`) was not used anywhere in v2.")
    (EVALUATION_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
