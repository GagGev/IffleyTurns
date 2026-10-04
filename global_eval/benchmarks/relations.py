"""Benchmark 2: curated relations (Orphanet siblings, shared causal gene, shared trial drug).

Each query disease ranks the whole catalogue; its positives are the diseases it shares the relation with.  As in
v2/benchmarks.py, the modalities that define a relation are hidden from every model (v1 hides the matching families).

IN-SAMPLE CAVEAT: v2's and v3's production models were fitted on these same relations (all diseases), so their
numbers are optimistic; v1 is unsupervised and unaffected.  Treat the gap between models as a measure of fit, not
generalisation.  The held-out versions are in v2/ (disease-level splits) and v3/ (forward in time).
"""
import numpy as np

import core

NAME = "relations"
INTRO = """Orphanet siblings, shared causal gene and shared trial drug, with the modalities that define each relation hidden.
**In-sample for v2 and v3**, whose production models were fitted on these relations; v1 is unsupervised."""
TITLE = "Curated relations (in-sample for v2/v3)"
KINDS = {"orphanet_siblings": "Orphanet siblings (ontology, name hidden)",
         "shared_causal_gene": "Shared causal gene (gene, pathway, Open Targets genes hidden)",
         "shared_drug": "Shared trial drug (drug, drug target, Open Targets genes hidden)"}
MODEL_ORDER = ["v1", "v2", "v2_drugfree", "v3_static"]


def run(tasks: dict) -> tuple[dict, str]:
    ids = tasks["catalogue"]; col = {d: i for i, d in enumerate(ids)}; split = tasks["split"]
    metrics, text = {}, []
    for kind, title in KINDS.items():
        info = tasks["relations"][kind]; queries = list(info["queries"])
        folder = core.DATA / "scores" / f"rel_{kind}"
        models = [m for m in MODEL_ORDER if (folder / f"{m}.npy").is_file()]
        per_model = {}
        for m in models:
            M = np.load(folder / f"{m}.npy"); rows = []
            for i, q in enumerate(queries):
                valid = np.ones(len(ids), bool); valid[col[q]] = False
                pos = np.zeros(len(ids), bool); pos[[col[p] for p in info["queries"][q]]] = True
                rows.append(core.relation_metrics(M[i][valid], pos[valid], np.random.default_rng(col[q])))
            per_model[m] = np.array(rows)
        test = np.array([split[q] == "test" for q in queries])
        out, rows_md = {}, []
        for m in models:
            vals = per_model[m]
            entry = {"n_queries": len(queries)}
            cells = []
            for j, metric in enumerate(core.RELATION_METRICS):
                mean, lo, hi = core.bootstrap(vals[:, j])
                entry[metric] = {"mean": mean, "ci": [lo, hi]}
                if metric in ("map", "mrr", "hits@10", "auroc"):
                    cells.append(f"{mean:.3f} [{lo:.3f}, {hi:.3f}]")
            if test.any():
                entry["map_test_split_queries"] = float(vals[test, 0].mean())
            out[m] = entry
            rows_md.append([core.MODELS[m][1], *cells, f"{vals[test, 0].mean():.3f}" if test.any() else "–"])
        diffs = {}
        for a, b in (("v2", "v1"), ("v3_static", "v2_drugfree")):
            if a in per_model and b in per_model:
                d, lo, hi = core.paired_difference(per_model[a][:, 0], per_model[b][:, 0])
                diffs[f"{a}-{b}"] = {"map_diff": d, "ci": [lo, hi]}
        metrics[kind] = {"n_pairs": info["n_pairs"], "n_eligible_queries": info["n_eligible_queries"],
                         "n_queries": len(queries), "models": out, "paired_map_differences": diffs}
        rows_hit = [[core.MODELS[m][1], f"{100 * out[m]['hits@10']['mean']:.1f}% [{100 * out[m]['hits@10']['ci'][0]:.1f}, {100 * out[m]['hits@10']['ci'][1]:.1f}]"]
                    for m in models]
        header = ["Model", "MAP [95% CI]", "MRR [95% CI]", "Hits@10 [95% CI]", "AUROC [95% CI]", "MAP, v2-test-split queries"]
        text.append(f"### {title}\n\n{info['n_pairs']:,} pairs; {len(queries)} of {info['n_eligible_queries']:,} eligible query diseases sampled (seed 0).\n\n"
                    + core.md_table(header, rows_md) + "\n\n"
                    + "Share of query diseases with at least one true partner in their top 10:\n\n"
                    + core.md_table(["Model", "Diseases with a hit in the top 10 [95% CI]"], rows_hit) + "\n\n"
                    + "\n".join(f"- MAP {core.MODELS[a.split('-')[0]][1]} − {core.MODELS[a.split('-')[1]][1]}: {v['map_diff']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]"
                                for a, v in diffs.items()) + "\n")
    return metrics, "\n".join(text)
