"""Benchmark 3: forward in time (v3's protocol), imported from ``v3/.data/evaluation``.

Models trained only on knowledge before a cutoff predict FDA/EMA orphan-designation relations formed afterwards. It
needs models retrained at each cutoff, so it is produced by ``v3/run_evaluation.py`` and imported here rather than
recomputed from the production weights (which were fitted through 2026-09 and would leak the answers).
"""
import shutil

import core

NAME = "forward_time"
TITLE = "Forward in time (v3 protocol)"
INTRO = """Cutoff 2018-01-01: models see relations known before it and rank the 9,525 nodes for 447 query diseases that gained
a relation between 2018 and 2026-09 (1,123 new relations). The v3 pipeline retrains v2's architecture on regulatory
relations (`v2_retrained`) because v2's own production weights were fitted on 2026 data. Imported from
`v3/.data/evaluation/summary.json`."""
SOURCE = core.ROOT / "v3" / ".data" / "evaluation"
SHOW = ["random", "degree", "adamic_adar", "drug_mechanism", "v1_weighted_jaccard", "v2_shipped", "v2_retrained",
        "gbm_static", "v3_neural", "v3_static", "v3_stacker", "stacker_no_history"]
LABEL = {"v1_weighted_jaccard": "v1 weighted Jaccard", "v2_shipped": "v2 shipped (production weights, drug-free)",
         "v2_retrained": "v2 architecture, retrained on pre-cutoff relations", "v3_static": "v3 static similarity",
         "v3_neural": "v3 neural only", "v3_stacker": "v3 stacker (all features; selected on validation)",
         "stacker_no_history": "v3 stacker without designation history", "gbm_static": "GBM on cosines",
         "drug_mechanism": "drug-mechanism overlap", "adamic_adar": "Adamic–Adar on known relations",
         "degree": "popularity (designation count + degree)", "random": "random"}


def _table(metrics: dict, queries: int) -> str:
    rows = []
    for name in SHOW:
        m = metrics.get(name)
        if not m:
            continue
        mp = m["map"]
        rows.append([LABEL.get(name, name), f"{mp['mean']:.3f} [{mp['ci_low']:.3f}, {mp['ci_high']:.3f}]",
                     f"{m['auc']['mean']:.3f}", f"{m['hits@10']['mean']:.3f}", f"{m['recall@50']['mean']:.3f}"])
    return core.md_table(["Model", "MAP [95% CI]", "AUROC", "Hits@10", "Recall@50"], rows)


def run(tasks: dict) -> tuple[dict, str]:
    summary_path = SOURCE / "summary.json"
    if not summary_path.is_file():
        return {}, "`v3/.data/evaluation/summary.json` not found. Run `python v3/run_evaluation.py` (about an hour on a laptop CPU)."
    summary = core.read_json(summary_path)
    out_dir = core.RESULTS / "forward_time"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("report.md", "per_query_metrics.csv", "map_by_model.png", "summary.json", "selected_config.json"):
        if (SOURCE / name).is_file():
            shutil.copy(SOURCE / name, out_dir / f"v3_{name}")

    metrics, text = {"end_of_data": summary["end_of_data"], "folds": {}}, []
    galleries = {"full": "All 9,525 nodes as candidates", "warm": "Only diseases already designated before the cutoff",
                 "full_approved": "All nodes; positives restricted to relations where both designations led to approval"}
    for fold in ("test", "validation"):
        f = summary["folds"][fold]; info = f["info"]
        metrics["folds"][fold] = {"info": {k: info[k] for k in ("cutoff", "until", "known_relations", "new_relations", "queries_full")},
                                  "metrics": {g: {m: {k: v for k, v in vals.items() if k in ("n_queries", "map", "auc", "hits@10", "recall@50", "mrr")}
                                                  for m, vals in f["metrics"][g].items()} for g in f["metrics"]},
                                  "global_ranking": f.get("global_ranking"), "stacker_importance": f.get("stacker_importance")}
        for gallery, title in galleries.items():
            if gallery in f["metrics"] and (fold == "test" or gallery == "full"):
                n = f["metrics"][gallery]["random"]["n_queries"]
                text.append(f"### {fold.capitalize()} fold (cutoff {info['cutoff']}): {title}, {n} queries\n\n{_table(f['metrics'][gallery], n)}\n")
        tests = f["paired_tests"]
        lines = []
        for a, versus in tests.items():
            for b, vals in versus.items():
                if b in ("v1_weighted_jaccard", "v2_retrained", "degree", "v2_shipped") and "map" in vals:
                    d = vals["map"]
                    lines.append(f"- {LABEL.get(a, a)} − {LABEL.get(b, b)}: MAP {d['mean_difference']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}], p = {d['p_wilcoxon_greater']:.1e}")
        if lines and fold == "test":
            text.append("Paired MAP differences on the test fold (bootstrap CI, one-sided Wilcoxon):\n\n" + "\n".join(lines) + "\n")

    base = out_dir / "trivial_baselines.txt"
    if base.is_file():
        text.append("### Trivial baselines on the same task (`v3_audit/audit_v3_baselines.py`)\n\n```\n" + base.read_text(encoding="utf-8").strip() + "\n```\n")
    text.append("Full v3 report: `results/forward_time/v3_report.md`; per-query metrics: `v3_per_query_metrics.csv`; plot: `v3_map_by_model.png`.\n")
    return metrics, "\n".join(text)
