"""Draw the evaluation-metrics infographic to ``global_eval/docs/metrics_infographic.png``.

The worked example is computed with ``core.relation_metrics``; benchmark results are read from
``results/metrics.json`` (run ``python global_eval/run.py`` first).

Usage:  python global_eval/make_infographic.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

import core

OUTPUT = Path(__file__).resolve().parent / "docs" / "metrics_infographic.png"
CATALOGUE = 7493
PARTNER_RANKS = (2, 5, 40)

TEXT, MUTED, WHITE = "#1f2933", "#52606d", "#ffffff"
GRAY = ("#7b8794", "#f5f7fa")
BLUE = ("#2f6db3", "#eaf2fb")
TEAL = ("#1f7a8c", "#e6f3f6")
ORANGE = ("#c46a1b", "#fdf1e6")
GREEN = ("#2b8a5b", "#e8f6ee")
PURPLE = ("#7b4fb3", "#f3eefb")
MODEL_COLORS = {"random": "#cbd2d9", "v1": "#d9a066", "v2": BLUE[0], "v3": "#52606d"}


def panel(ax, x, y, w, h, palette, fill=None, dashed=False):
    edge, face = palette
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", linewidth=1.4,
            edgecolor=edge, facecolor=fill or face, linestyle=(0, (4, 3)) if dashed else "-",
        )
    )


def worked_example() -> dict[str, float]:
    n = CATALOGUE - 1
    positives = np.zeros(n, dtype=bool)
    positives[[r - 1 for r in PARTNER_RANKS]] = True
    ap, rr, hit10, p10, ndcg10, auc = core.relation_metrics(-np.arange(n, dtype=float), positives, np.random.default_rng(0))
    return {
        "map": ap, "mrr": rr, "hits@10": hit10, "precision@10": p10, "ndcg@10": ndcg10, "auroc": auc,
        "recall@50": float(np.mean([r <= 50 for r in PARTNER_RANKS])),
        "percentile": 1 - (PARTNER_RANKS[-1] - 1) / (CATALOGUE - 2),
        "median_rank": float(np.median(PARTNER_RANKS)),
    }


def ranked_list(ax, x, top):
    rows = [(r, r in PARTNER_RANKS) for r in range(1, 11)] + [(None, False), (40, True), (None, False), (CATALOGUE - 1, False)]
    found = 0
    for k, (rank, partner) in enumerate(rows):
        y = top - k * 3.0 - 2.5
        if rank is None:
            ax.text(x + 22, y + 1.25, "⋮", fontsize=11, color=MUTED, ha="center", va="center")
            continue
        edge, face = GREEN if partner else GRAY
        ax.text(x + 5.2, y + 1.25, f"#{rank:,}", fontsize=8.6, color=edge if partner else MUTED, ha="right", va="center",
                fontweight="bold" if partner else "normal")
        ax.add_patch(FancyBboxPatch((x + 6.2, y), 33.8, 2.5, boxstyle="round,pad=0,rounding_size=0.6", linewidth=1.0,
                                    edgecolor=edge, facecolor=face if not partner else "#cfeedd"))
        if partner:
            found += 1
            ax.text(x + 7.4, y + 1.25, "true partner", fontsize=8.6, color=GREEN[0], fontweight="bold", va="center")
            ax.text(x + 39.0, y + 1.25, f"precision {found}/{rank}", fontsize=8.0, color=GREEN[0], ha="right", va="center")
        else:
            ax.text(x + 7.4, y + 1.25, "unrelated disease", fontsize=8.4, color=MUTED, va="center")
    bracket_top, bracket_bottom = top - 0.1, top - 9 * 3.0 - 2.4
    ax.plot([x + 40.8, x + 41.6, x + 41.6, x + 40.8], [bracket_top, bracket_top, bracket_bottom, bracket_bottom], color=BLUE[0], linewidth=1.2)
    ax.text(x + 42.6, (bracket_top + bracket_bottom) / 2, "top 10", fontsize=8.6, color=BLUE[0], rotation=90, ha="center", va="center", fontweight="bold")


def metric_card(ax, x, y, w, h, name, value, question, example, used, palette):
    panel(ax, x, y, w, h, palette, fill=WHITE)
    top = y + h
    ax.text(x + 1.4, top - 1.5, name, fontsize=10.5, fontweight="bold", color=palette[0], va="top")
    ax.text(x + w - 1.4, top - 1.2, value, fontsize=13, fontweight="bold", color=palette[0], ha="right", va="top")
    ax.text(x + 1.4, top - 5.2, question, fontsize=8.4, color=TEXT, va="top", linespacing=1.4)
    ax.text(x + 1.4, top - 10.4, example, fontsize=8.0, color=GREEN[0], va="top", style="italic", linespacing=1.35)
    ax.text(x + 1.4, y + 1.2, used, fontsize=7.8, color=MUTED, va="bottom", linespacing=1.3)


def benchmark_card(ax, x, y, w, h, palette, title, query, correct, fair, metrics, chart_title, bars):
    panel(ax, x, y, w, h, palette)
    top = y + h
    ax.text(x + 1.6, top - 1.6, title, fontsize=10.5, fontweight="bold", color=palette[0], va="top")
    for k, (label, body) in enumerate((("Query", query), ("Correct", correct), ("Fair test", fair))):
        yy = top - 5.5 - k * 4.3
        ax.text(x + 1.6, yy, label, fontsize=8.0, fontweight="bold", color=MUTED, va="top")
        ax.text(x + 9.0, yy, body, fontsize=8.2, color=TEXT, va="top", linespacing=1.35)
    ax.text(x + 1.6, top - 18.9, metrics, fontsize=8.2, color=palette[0], va="top", linespacing=1.35, fontweight="bold")

    largest = max(v for _, _, v in bars)
    base = y + 1.6
    ax.text(x + 1.6, base + len(bars) * 1.7 + 0.3, chart_title, fontsize=8.0, color=MUTED, va="center", style="italic")
    for k, (label, model, value) in enumerate(bars):
        yy = base + (len(bars) - 1 - k) * 1.7
        ax.text(x + 8.4, yy, label, fontsize=8.0, color=TEXT, ha="right", va="center")
        length = 21.0 * value / largest
        ax.add_patch(Rectangle((x + 9.2, yy - 0.55), max(length, 0.15), 1.1, linewidth=0, facecolor=MODEL_COLORS[model]))
        ax.text(x + 9.2 + length + 0.6, yy, f"{value:.3f}", fontsize=7.8, color=TEXT, va="center")


def main() -> None:
    metrics = core.read_json(core.RESULTS / "metrics.json")
    ex = worked_example()
    paper = metrics["paper_pairs"]["strata"]["no_curated_relation"]["models"]
    siblings = metrics["relations"]["orphanet_siblings"]["models"]
    forward = metrics["forward_time"]["folds"]["test"]["metrics"]["full"]
    forward_info = metrics["forward_time"]["folds"]["test"]["info"]
    symptoms = metrics["symptom_retrieval"]["k5"]
    n_symptom_queries = symptoms["v2"]["n"]

    fig = plt.figure(figsize=(16, 11))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 110)
    ax.axis("off")

    ax.text(2, 107.5, "How the rare-disease models are scored", fontsize=17, fontweight="bold", color=TEXT, va="top")
    ax.text(
        2, 102.8,
        f"Every benchmark asks one question: give the model a disease, let it rank the other {CATALOGUE - 1:,} diseases, "
        "and see where the true partners land.",
        fontsize=10, color=MUTED, va="top",
    )

    # 1 Worked example
    ax.text(2, 97.5, "1  One query, ranked", fontsize=11.5, fontweight="bold", color=MUTED, va="center")
    ranked_list(ax, 2, 93.5)

    # 2 Metric cards
    ax.text(50, 97.5, "2  What each metric asks, computed on this query", fontsize=11.5, fontweight="bold", color=MUTED, va="center")
    p = PARTNER_RANKS
    cards = [
        ("Hits@10", "yes", "Is any true partner in the top 10?\nAlso Hits@50 and Hits@100", f"first partner at rank {p[0]} ≤ 10",
         "all four benchmarks", BLUE),
        ("MRR", f"{ex['mrr']:.2f}", "How high is the first true partner?\n1 / its rank, averaged over queries", f"1 / {p[0]}",
         "paper pairs, relations, symptoms", BLUE),
        ("MAP", f"{ex['map']:.3f}", "Are all partners near the top?\nPrecision at each partner, averaged",
         f"(1/{p[0]} + 2/{p[1]} + 3/{p[2]}) / 3", "relations, forward time (headline)", BLUE),
        ("Precision@10", f"{ex['precision@10']:.2f}", "What share of the top 10 is a true\npartner?",
         f"2 of 10; Recall@50 = {ex['recall@50']:.0f}\n(share of partners in the top 50)", "relations, forward time", BLUE),
        ("nDCG@10", f"{ex['ndcg@10']:.2f}", "Top-10 hits, each discounted by\nlog2(rank + 1), versus a perfect list",
         "(1/log2 3 + 1/log2 6) / best case", "relations, v2's own evaluation", GRAY),
        ("AUROC", f"{ex['auroc']:.3f}", "Does a random partner outrank a\nrandom unrelated disease?",
         f"yes for {100 * ex['auroc']:.1f}% of such pairs;\nrandom ranking = 0.5", "relations, forward time", GRAY),
        ("Partner percentile", f"{ex['percentile']:.3f}", "Share of the catalogue ranked\nbelow the partner",
         f"partner at rank {p[2]}: 1 − {p[2] - 1} / {CATALOGUE - 2:,}", "paper pairs (mean over pairs)", GRAY),
        ("Median rank", f"{ex['median_rank']:.0f}", "The middle partner rank over all\nqueries",
         f"queries with partners at {p[0]}, {p[1]}, {p[2]} → {ex['median_rank']:.0f}", "paper pairs, symptoms", GRAY),
    ]
    w, h, gap = 25.5, 20.0, 2.0
    for k, (name, value, question, example, used, palette) in enumerate(cards):
        cx = 50 + (k % 4) * (w + gap)
        cy = 73.5 if k < 4 else 51.5
        metric_card(ax, cx, cy, w, h, name, value, question, example, f"used in {used}", palette)

    # Insight
    panel(ax, 2, 43.8, 156, 5.6, ORANGE)
    ax.text(
        3.6, 46.6,
        f"Why several metrics?  The same ranking scores AUROC {ex['auroc']:.3f} and percentile {ex['percentile']:.3f}, but MAP only "
        f"{ex['map']:.3f}. With {CATALOGUE - 1:,} candidates, almost any ranking beats most unrelated diseases,\n"
        "so AUROC and percentile look high for every model. MAP and Hits@10 measure what a researcher actually sees at the top "
        "of the list, so they are the headline numbers.",
        fontsize=9.0, color=TEXT, va="center", linespacing=1.45,
    )

    # 3 Benchmarks
    ax.text(2, 41.2, "3  How each benchmark uses them", fontsize=11.5, fontweight="bold", color=MUTED, va="center")
    bw, bgap, by, bh = 37.5, 2.0, 6.5, 32.5
    benchmarks = [
        (TEAL, "Paper-stated pairs",
         "one disease of a pair that a\npublished paper calls related",
         f"its paper partner; {metrics['paper_pairs']['n_pairs']} pairs,\nscored in both directions",
         f"never a training label; the {metrics['paper_pairs']['strata']['no_curated_relation']['n']} pairs\nwithout a curated relation are cleanest",
         "partner percentile, MRR, Hits@10/50/100,\nmedian rank; Spearman vs stated similarity",
         "Hits@10, pairs without a curated relation",
         [("v1", "v1", paper["v1"]["hits@10"]), ("v2", "v2", paper["v2"]["hits@10"]), ("v3", "v3", paper["v3_static"]["hits@10"])]),
        (ORANGE, "Curated relations",
         f"{metrics['relations']['orphanet_siblings']['n_queries']} sampled diseases per relation\ntype",
         "Orphanet siblings, shared causal\ngene or shared trial drug partners",
         "the profiles that define the relation\nare hidden; in-sample for v2 and v3",
         "MAP (headline), MRR, Hits@10, AUROC",
         "MAP, Orphanet siblings",
         [("v1", "v1", siblings["v1"]["map"]["mean"]), ("v2", "v2", siblings["v2"]["map"]["mean"]),
          ("v3", "v3", siblings["v3_static"]["map"]["mean"])]),
        (GREEN, "Forward in time",
         f"{forward_info['queries_full']} diseases that gained an\norphan-drug relation after 2018",
         f"the {forward_info['new_relations']:,} relations formed between\n2018 and Sep 2026",
         "models see only what was known\nbefore 2018 (retrained at the cutoff)",
         "MAP (headline), AUROC, Hits@10,\nRecall@50",
         "MAP, test fold (v3 = forecast model)",
         [("random", "random", forward["random"]["map"]["mean"]), ("v1", "v1", forward["v1_weighted_jaccard"]["map"]["mean"]),
          ("v2", "v2", forward["v2_retrained"]["map"]["mean"]), ("v3", "v3", forward["v3_stacker"]["map"]["mean"])]),
        (PURPLE, "Symptom-only retrieval",
         f"3, 5 or 10 of a disease's own symptoms\nand nothing else ({n_symptom_queries} diseases)",
         "the disease itself; relaxed: it or an\nOrphanet sibling in the top 10",
         "no name, genes or text: what a\nclinician has from a patient",
         "share in the top 1/10/20, median rank,\nMRR, relaxed top 10",
         "share in the top 10, 5 symptoms",
         [("v1", "v1", symptoms["v1"]["top10"]), ("v2", "v2", symptoms["v2"]["top10"]), ("v3", "v3", symptoms["v3_static"]["top10"])]),
    ]
    for k, args in enumerate(benchmarks):
        benchmark_card(ax, 2 + k * (bw + bgap), by, bw, bh, *args)

    ax.text(
        2, 4.6,
        "Every number is a mean over query diseases with a 95% bootstrap interval (2,000 resamples); models are compared by paired "
        "differences on the same queries.\nUnlabelled pairs count as wrong, so every score is a lower bound. Relations and symptom "
        "queries are in-sample for v2 and v3; paper pairs and forward time are not.",
        fontsize=8.6, color=MUTED, va="top", linespacing=1.45,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=160, facecolor=WHITE)
    print(OUTPUT)


if __name__ == "__main__":
    main()
