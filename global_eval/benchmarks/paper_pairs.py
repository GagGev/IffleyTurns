"""Benchmark 1: disease pairs that published papers describe as related.

Independent of every model's training labels, apart from pairs that also have a curated relation (flagged and
reported separately).  For each pair and model, the partner's rank among all catalogue diseases is taken in both
directions.  Also: Spearman with the papers' stated similarity, and AUROC separating pairs papers call "similar"
from "related but distinct".
"""
import numpy as np
from scipy.stats import rankdata, spearmanr

import core

NAME = "paper_pairs"
INTRO = """Disease pairs that published papers describe as related or similar.
Nothing about them was used to train any model, except where a pair is also a curated relation (reported separately).
For each pair, the partner's rank among the 7,493 catalogue diseases, in both directions."""
TITLE = "Paper-stated disease pairs"
MODEL_ORDER = ["v1", "v1_drugfree", "v2", "v2_drugfree", "v3_static", "v3_forecast"]


def _auc(score, label):
    if label.all() or (~label).all():
        return float("nan")
    r = rankdata(score)
    n1, n0 = label.sum(), (~label).sum()
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def run(tasks: dict) -> tuple[dict, str]:
    pairs = tasks["paper_pairs"]
    ids = tasks["catalogue"]; col = {d: i for i, d in enumerate(ids)}; n = len(ids)
    queries = sorted({p[k] for p in pairs for k in ("a", "b")}); row = {q: i for i, q in enumerate(queries)}
    folder = core.DATA / "scores" / "paper_pairs"
    models = [m for m in MODEL_ORDER if (folder / f"{m}.npy").is_file()]
    ranks, sym = {}, {}
    for m in models:
        M = np.load(folder / f"{m}.npy")
        r = np.zeros((len(pairs), 2)); s = np.zeros(len(pairs))
        for k, p in enumerate(pairs):
            r[k, 0] = core.rank_of_partner(M[row[p["a"]]], col[p["a"]], col[p["b"]])
            r[k, 1] = core.rank_of_partner(M[row[p["b"]]], col[p["b"]], col[p["a"]])
            s[k] = (M[row[p["a"]], col[p["b"]]] + M[row[p["b"]], col[p["a"]]]) / 2
        ranks[m], sym[m] = r, s

    stated = np.array([p["stated"] for p in pairs]); similar = np.array([p["similar"] for p in pairs])
    curated = np.array([bool(p["curated"]) for p in pairs])
    percentile = {m: (1 - (ranks[m] - 1) / (n - 2)).mean(axis=1) for m in models}   # per pair, both directions

    def table(selector):
        idx = np.where(selector)[0]
        out, rows = {}, []
        for m in models:
            r = ranks[m][idx]
            mp, lo, hi = core.bootstrap(percentile[m][idx])
            vals = {"mean_percentile": mp, "ci": [lo, hi], "mrr": float((1 / r).mean()),
                    "hits@10": float((r <= 10).mean()), "hits@50": float((r <= 50).mean()),
                    "hits@100": float((r <= 100).mean()), "median_rank": float(np.median(r))}
            out[m] = vals
            rows.append([core.MODELS[m][1], f"{mp:.3f} [{lo:.3f}, {hi:.3f}]", f"{vals['mrr']:.3f}", f"{vals['hits@10']:.3f}",
                         f"{vals['hits@50']:.3f}", f"{vals['hits@100']:.3f}", f"{vals['median_rank']:.0f}"])
        header = ["Model", "Mean percentile of partner [95% CI]", "MRR", "Hits@10", "Hits@50", "Hits@100", "Median rank"]
        return out, core.md_table(header, rows), len(idx)

    metrics, text = {"n_pairs": len(pairs), "strata": {}}, []
    strata = {
        "all": np.ones(len(pairs), bool),
        "no_curated_relation": ~curated,
        "curated_relation": curated,
    }
    for dim in ("phenotype", "genes", "pathways", "treatment"):
        sel = np.array([dim in p["dims"] for p in pairs])
        if sel.sum() >= 20:
            strata[f"dimension_{dim}"] = sel
    titles = {"all": "All pairs", "no_curated_relation": "Pairs with no curated relation (not a training label for v2/v3)",
              "curated_relation": "Pairs with a curated relation (inside v2/v3 training labels)"}
    for key, sel in strata.items():
        out, md, count = table(sel)
        metrics["strata"][key] = {"n": count, "models": out}
        text.append(f"### {titles.get(key, 'Papers’ stated dimension: ' + key.replace('dimension_', ''))} ({count} pairs)\n\n{md}\n")

    rows, tracking = [], {}
    for m in models:
        rho, rlo, rhi = core.bootstrap_stat(lambda i: spearmanr(sym[m][i], stated[i])[0], len(pairs))
        a, alo, ahi = core.bootstrap_stat(lambda i: _auc(sym[m][i], similar[i]), len(pairs))
        tracking[m] = {"spearman": rho, "spearman_ci": [rlo, rhi], "auroc_similar": a, "auroc_ci": [alo, ahi]}
        rows.append([core.MODELS[m][1], f"{rho:.3f} [{rlo:.3f}, {rhi:.3f}]", f"{a:.3f} [{alo:.3f}, {ahi:.3f}]"])
    metrics["tracking"] = tracking
    text.append("### Does the score track the papers’ stated similarity?\n\n"
                + core.md_table(["Model", "Spearman with stated similarity [95% CI]", "AUROC similar vs related-but-distinct [95% CI]"], rows)
                + f"\n\n{int(similar.sum())} pairs are “similar”, {int((~similar).sum())} “related but distinct”.\n")

    comparisons = [("v2", "v1"), ("v3_static", "v2_drugfree"), ("v3_static", "v2"), ("v3_forecast", "v3_static")]
    lines, diffs = [], {}
    for a, b in comparisons:
        if a not in percentile or b not in percentile:
            continue
        for label, sel in (("all pairs", strata["all"]), ("no curated relation", strata["no_curated_relation"])):
            d, lo, hi = core.paired_difference(percentile[a][sel], percentile[b][sel])
            diffs[f"{a}-{b}:{label}"] = {"diff": d, "ci": [lo, hi]}
            lines.append(f"- {core.MODELS[a][1]} − {core.MODELS[b][1]} ({label}): {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    metrics["paired_differences_mean_percentile"] = diffs
    text.append("### Paired differences in mean percentile of the partner (95% bootstrap CI)\n\n" + "\n".join(lines) + "\n")

    per_pair = [{**{k: p[k] for k in ("a", "b", "stated", "similar", "dims", "curated", "papers")},
                 **{f"rank_{m}": [float(x) for x in ranks[m][k]] for m in models}} for k, p in enumerate(pairs)]
    core.write_json(core.RESULTS / "paper_pairs_per_pair.json", per_pair)
    return metrics, "\n".join(text)
