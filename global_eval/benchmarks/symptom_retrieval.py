"""Benchmark 4: symptom-only retrieval (the patient-view scenario).

A query is a handful of a disease's own phenotypes (3, 5 or 10 of them; in one variant five plus one unrelated
term) and nothing else: no name, genes or description. The model ranks the whole catalogue and we ask whether the
disease itself, or a close relative, comes back near the top. This is a cold-start test: the query record is not in
the model's training set as such, though the disease it was drawn from is in the catalogue (that is the target).

Reported per variant: the share of queries whose true disease is in the top 1 / 10 / 20, the median rank of the
true disease, and a relaxed hit: the top 10 contains the disease itself or one of its Orphanet siblings.
"""
import numpy as np

import core

NAME = "symptom_retrieval"
TITLE = "Symptom-only retrieval (patient scenario)"
INTRO = """A query is 3, 5 or 10 of a disease's own phenotypes, or 5 plus one unrelated one, and nothing else. The model ranks
7,493 diseases. v1 is the Jaccard of raw HPO sets (v1 has no other way to use a bare symptom list); v2 and v3 encode the
symptoms as a new disease. v3 here is the production weights; the v2 production model also saw these diseases' profiles,
so the *rate of finding the true disease* is optimistic for both. The relaxed hit and the comparison between models are
the informative parts."""
MODEL_ORDER = ["v1", "v2", "v3_static"]
VARIANTS = {"k3": "3 symptoms", "k5": "5 symptoms", "k10": "10 symptoms", "k5_noise1": "5 symptoms + 1 unrelated"}


def run(tasks: dict) -> tuple[dict, str]:
    queries = tasks.get("symptom_queries")
    folder = core.DATA / "scores" / "symptoms"
    if not queries or not folder.is_dir():
        return {}, "Not scored. Run `python global_eval/run.py --rescore`."
    ids = tasks["catalogue"]; col = {d: i for i, d in enumerate(ids)}
    models = [m for m in MODEL_ORDER if (folder / f"{m}.npy").is_file()]
    matrices = {m: np.load(folder / f"{m}.npy") for m in models}
    metrics, rows_md = {}, []
    for variant, label in VARIANTS.items():
        idx = [i for i, q in enumerate(queries) if q["variant"] == variant]
        metrics[variant] = {}
        for m in models:
            ranks, relaxed = [], []
            for i in idx:
                q = queries[i]; s = matrices[m][i].astype(np.float64)
                truth = col[q["disease"]]
                rank = 1 + (s > s[truth]).sum() + 0.5 * ((s == s[truth]).sum() - 1)
                ranks.append(rank)
                top10 = np.argsort(-s, kind="stable")[:10]
                family = {truth, *(col[x] for x in q["siblings"] if x in col)}
                relaxed.append(any(j in family for j in top10))
            ranks = np.array(ranks)
            e = {"n": len(idx), "top1": float((ranks <= 1).mean()), "top10": float((ranks <= 10).mean()),
                 "top20": float((ranks <= 20).mean()), "median_rank": float(np.median(ranks)),
                 "mrr": float((1 / ranks).mean()), "relaxed_top10": float(np.mean(relaxed))}
            _, lo, hi = core.bootstrap((ranks <= 10).astype(float))
            e["top10_ci"] = [lo, hi]
            metrics[variant][m] = e
            rows_md.append([label, core.MODELS[m][1] if m in core.MODELS else m, str(len(idx)), f"{100 * e['top1']:.0f}%",
                            f"{100 * e['top10']:.0f}% [{100 * lo:.0f}, {100 * hi:.0f}]", f"{100 * e['top20']:.0f}%",
                            f"{e['median_rank']:.0f}", f"{100 * e['relaxed_top10']:.0f}%"])
    table = core.md_table(["Query", "Model", "Queries", "True disease top 1", "top 10 [95% CI]", "top 20", "Median rank",
                           "Disease or a sibling in top 10"], rows_md)
    return metrics, table + "\n"
