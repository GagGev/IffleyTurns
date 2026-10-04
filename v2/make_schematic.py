"""Draw the v2 model schematic to ``v2/docs/model_schematic.png``.

Numbers are read from ``run_evaluation.py`` (``evaluation/metrics.json``) and
``build_graph.py`` (``graph/graph_summary.json``), so run those first.

Usage:  python v2/make_schematic.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from common import EVALUATION_DIR, GRAPH_DIR
from modalities import MODALITIES

OUTPUT = Path(__file__).resolve().parent / "docs" / "model_schematic.png"

TEXT, MUTED = "#1f2933", "#52606d"
GRAY = ("#7b8794", "#f5f7fa")
BLUE = ("#2f6db3", "#eaf2fb")
ORANGE = ("#c46a1b", "#fdf1e6")
GREEN = ("#2b8a5b", "#e8f6ee")
WHITE = "#ffffff"

BENCHMARKS = (
    ("orphanet_siblings", "Orphanet siblings", "siblings"),
    ("shared_causal_gene", "Shared causal gene", "shared gene"),
    ("shared_drug", "Shared trial drug", "shared drug"),
)
TEST_MODELS = (("random", "random", "#cbd2d9"), ("v1_weighted_jaccard", "v1 weighted Jaccard", "#e6b17e"), ("logistic_fusion", "v2", GREEN[0]))


def load() -> tuple[dict, dict]:
    paths = (EVALUATION_DIR / "metrics.json", GRAPH_DIR / "graph_summary.json")
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"{path} not found. Run `python v2/run_evaluation.py` and `python v2/build_graph.py` first.")
    metrics, graph = (json.loads(path.read_text(encoding="utf-8")) for path in paths)
    return metrics, graph


def box(ax, x, y, w, h, palette, title=None, body=None, footer=None, dashed=False, fill=None, size=9.2):
    edge, face = palette
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0,rounding_size=1.2",
            linewidth=1.5, edgecolor=edge, facecolor=fill or face,
            linestyle=(0, (4, 3)) if dashed else "-",
        )
    )
    top = y + h - 1.6
    if title:
        ax.text(x + 1.6, top, title, fontsize=11, fontweight="bold", color=edge, va="top")
        top -= 4.0
    if body:
        ax.text(x + 1.6, top, body, fontsize=size, color=TEXT, va="top", linespacing=1.5)
    if footer:
        ax.text(x + 1.6, y + 1.5, footer, fontsize=size, fontweight="bold", color=edge, va="bottom", linespacing=1.4)


def arrow(ax, start, end, color, dashed=False, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start, end,
            arrowstyle="-|>", mutation_scale=16, linewidth=1.8, color=color,
            linestyle=(0, (4, 3)) if dashed else "-",
            connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2,
        )
    )


def note(ax, x, y, text, color, ha="center", va="center", size=8.4):
    ax.text(x, y, text, fontsize=size, color=color, ha=ha, va=va, style="italic", linespacing=1.3)


def chips(ax, x, y, names, edge, per_row=3, w=10.0, h=2.8, gap=0.8, hidden=False, size=8.4):
    for k, name in enumerate(names):
        cx = x + (k % per_row) * (w + gap)
        cy = y - (k // per_row) * (h + gap)
        ax.add_patch(
            FancyBboxPatch(
                (cx, cy), w, h, boxstyle="round,pad=0,rounding_size=0.8", linewidth=1.0,
                edgecolor=edge, facecolor=WHITE, linestyle=(0, (2, 2)) if hidden else "-",
            )
        )
        ax.text(cx + w / 2, cy + h / 2, name, fontsize=size, color=MUTED if hidden else TEXT, ha="center", va="center")


def weight_bars(ax, x, y, coefficients, length=21.0, step=1.85):
    rows = sorted(coefficients, key=lambda c: -c["similarity_weight"])
    largest = max(c["similarity_weight"] for c in rows)
    for k, row in enumerate(rows):
        yy = y - k * step
        weight = row["similarity_weight"]
        ax.text(x - 0.8, yy, row["modality"], fontsize=8.4, color=TEXT, ha="right", va="center")
        bar = length * weight / largest
        if bar > 0:
            ax.add_patch(Rectangle((x, yy - 0.6), bar, 1.2, linewidth=0, facecolor=BLUE[0], alpha=0.8))
        ax.text(x + bar + 0.6, yy, f"{weight:.2f}" if weight > 0 else "0", fontsize=8.0, color=MUTED, va="center")


def evaluation_strip(ax, metrics, n_diseases):
    splits = metrics["data"]["split_sizes"]
    ax.text(2, 17.5, "Disease-held-out evaluation", fontsize=11, fontweight="bold", color=TEXT, va="center")
    ax.text(
        2, 14.6,
        "Diseases are split 60/20/20 by a stable\nhash, so test diseases are new to every\n"
        f"encoder and weight. The released model\nis refit on all {n_diseases:,} diseases.",
        fontsize=8.6, color=MUTED, va="top", linespacing=1.4,
    )

    x, y, width = 40.0, 12.0, 62.0
    segments = (
        ("train", splits["train"], BLUE, "fit encoders and fusion weights"),
        ("validation", splits["validation"], ORANGE, "choose C and\nthe fusion type"),
        ("test", splits["test"], GREEN, "report the\nresults"),
    )
    total = sum(n for _, n, _, _ in segments)
    for name, n, (edge, face), purpose in segments:
        w = width * n / total
        ax.add_patch(Rectangle((x, y), w, 3.2, linewidth=1.0, edgecolor=WHITE, facecolor=edge, alpha=0.85))
        ax.text(x + w / 2, y + 1.6, name, fontsize=8.4, color=WHITE, fontweight="bold", ha="center", va="center")
        ax.text(x + w / 2, y - 1.0, f"{n:,} diseases\n{purpose}", fontsize=8.0, color=edge, ha="center", va="top", linespacing=1.35)
        x += w

    test = metrics["test"]
    x0, base, height = 108.0, 4.6, 9.5
    ax.text(x0, 18.6, "Test MAP, with each relation's own profiles hidden", fontsize=9.2, fontweight="bold", color=GREEN[0], va="center")
    lx = x0
    for _, label, color in TEST_MODELS:
        ax.add_patch(Rectangle((lx, 15.9), 1.2, 1.2, linewidth=0, facecolor=color))
        ax.text(lx + 1.7, 16.5, label, fontsize=8.0, color=MUTED, va="center")
        lx += 3.2 + 0.62 * len(label)
    values = {b: {m: test[b]["all"]["models"][m]["map"]["mean"] for m, _, _ in TEST_MODELS} for b, _, _ in BENCHMARKS}
    largest = max(v for row in values.values() for v in row.values())
    bar_w, gap = 3.6, 0.5
    for g, (key, _, short) in enumerate(BENCHMARKS):
        gx = x0 + 2.0 + g * 17.0
        for k, (model, _, color) in enumerate(TEST_MODELS):
            bx = gx + k * (bar_w + gap)
            h = height * values[key][model] / largest
            ax.add_patch(Rectangle((bx, base), bar_w, h, linewidth=0, facecolor=color))
            ax.text(bx + bar_w / 2, base + h + 0.4, f"{values[key][model]:.3f}", fontsize=7.2, color=TEXT, ha="center", va="bottom")
        ax.plot([gx - 0.4, gx + 3 * bar_w + 2 * gap + 0.4], [base, base], color=MUTED, linewidth=0.8)
        ax.text(gx + (3 * bar_w + 2 * gap) / 2, base - 0.7, short, fontsize=8.0, color=MUTED, ha="center", va="top")


def main() -> None:
    metrics, graph = load()
    benchmarks = metrics["data"]["benchmarks"]
    model = graph["model"]
    n_diseases = graph["nodes"]
    macro = metrics["validation_selection"]["macro"]
    selected = metrics["selected"]

    fig = plt.figure(figsize=(16, 10.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 106)
    ax.axis("off")

    ax.text(2, 103.2, "v2: how the rare-disease similarity graph is built", fontsize=17, fontweight="bold", color=TEXT, va="top")
    ax.text(
        2, 98.6,
        "Similarity is a weighted sum of twelve kinds of evidence. The weights are learned from curated disease relations, "
        "each time hiding the evidence that relation was built from.",
        fontsize=10, color=MUTED, va="top",
    )
    c1, c2, c3, c4 = 2, 38, 78, 125
    w1, w2, w3, w4 = 31, 34, 41, 33
    for x, label in ((c1, "1  Inputs"), (c2, "2  Profiles and pairs"), (c3, "3  Model"), (c4, "4  Outputs")):
        ax.text(x, 93.0, label, fontsize=11.5, fontweight="bold", color=MUTED, va="center")

    # 1 Inputs
    box(
        ax, c1, 59, w1, 32, GRAY, "Disease databases",
        "Orphanet: classification, names,\n  descriptions, inheritance, onset,\n  prevalence\n"
        "HPO phenotypes, Mondo ontology\nGenes: Orphanet, HPO, ClinGen,\n  ClinVar, Open Targets\n"
        "Reactome pathways of those genes\nDrugs: Open Targets / ChEMBL,\n  FDA and EMA orphan designations",
        footer=f"{n_diseases:,} Orphanet diseases",
    )
    related = sum(benchmarks[key]["related_pairs"] for key, _, _ in BENCHMARKS)
    box(
        ax, c1, 24, w1, 29, ORANGE, "Curated relations",
        "Orphanet classification siblings\n  and subtypes\nShared causal gene (Orphanet)\n"
        "Shared trial or approved drug,\n  phase 2 or later (Open Targets)\n\n"
        "Used only as labels: to fit the\nweights and to test them",
        footer=f"{related:,} related pairs",
    )

    # 2 Profiles and pairs
    box(ax, c2, 59, w2, 32, BLUE, "Twelve disease profiles", fill=GRAY[1])
    chips(ax, c2 + 1.6, 82.2, MODALITIES, BLUE[0])
    ax.text(
        c2 + 1.6, 69.4,
        "each a weighted sparse vector; IDF,\ninformation content and text\nvocabulary learned on training\ndiseases only",
        fontsize=8.8, color=TEXT, va="top", linespacing=1.45,
    )
    box(ax, c2, 24, w2, 29, ORANGE, "Masked training pairs")
    for k, (key, label, _) in enumerate(BENCHMARKS):
        yy = 47.0 - k * 6.2
        ax.text(c2 + 1.6, yy, f"{label}: {benchmarks[key]['related_pairs']:,} pairs", fontsize=8.8, color=TEXT, va="center")
        ax.text(c2 + 1.6, yy - 3.0, "hides", fontsize=8.0, color=MUTED, va="center", style="italic")
        chips(ax, c2 + 6.4, yy - 4.3, benchmarks[key]["masked_modalities"], GRAY[0], w=8.2, h=2.6, gap=0.6, hidden=True, size=8.0)
    ax.text(
        c2 + 1.6, 25.4,
        f"+ {selected['negatives_per_positive']} random unrelated pairs per related\n  pair; each relation type weighted equally",
        fontsize=8.2, color=MUTED, va="bottom", linespacing=1.35,
    )

    # 3 Model
    box(ax, c3, 24, w3, 67, BLUE, "Logistic fusion: the similarity model")
    ax.text(
        c3 + 1.6, 85.4,
        "For diseases A and B, each profile m gives\n"
        r"   $s_m$   cosine similarity, from 0 to 1" "\n"
        r"   $a_m$   1 if both diseases have profile m",
        fontsize=9.2, color=TEXT, va="top", linespacing=1.6,
    )
    ax.add_patch(
        FancyBboxPatch((c3 + 1.6, 72.0), w3 - 3.2, 6.4, boxstyle="round,pad=0,rounding_size=0.8", linewidth=1.2, edgecolor=BLUE[0], facecolor=WHITE)
    )
    ax.text(c3 + w3 / 2, 76.4, r"$\mathrm{score} = b + \sum_m w_m\, s_m + \sum_m v_m\, a_m$", fontsize=12.5, color=BLUE[0], ha="center", va="center")
    ax.text(c3 + w3 / 2, 73.6, r"$w_m \geq 0$: shared evidence can only raise the score", fontsize=8.4, color=MUTED, ha="center", va="center")
    ax.text(c3 + 1.6, 69.8, r"Learned weights $w_m$", fontsize=9.2, fontweight="bold", color=TEXT, va="center")
    ax.text(c3 + 18.4, 69.8, f"(fit on all {n_diseases:,} diseases)", fontsize=8.4, color=MUTED, va="center")
    weight_bars(ax, c3 + 13.2, 67.0, model["coefficients"])
    ax.text(
        c3 + 1.6, 44.2,
        "Correlated profiles share one weight: pathway\nand Open Targets genes absorb gene's. A new\n"
        "disease without them gets a refit (column 4).",
        fontsize=8.4, color=MUTED, va="top", linespacing=1.4,
    )
    ax.text(
        c3 + 1.6, 36.4,
        "The score is additive, so every edge's explanation\n"
        r"is exact: profile m contributed $w_m\, s_m$.",
        fontsize=8.8, color=TEXT, va="top", linespacing=1.5,
    )
    ax.text(
        c3 + 1.6, 25.5,
        f"Weighted L2 logistic regression, C = {model['config']['logistic_params']['C']:g}\n"
        f"Validation MAP {macro[selected['logistic']]:.3f}; boosted trees\n"
        f"{macro['gbm_fusion']:.3f}, uniform mean {macro[selected['uniform']]:.3f}",
        fontsize=9.0, fontweight="bold", color=BLUE[0], va="bottom", linespacing=1.4,
    )

    # 4 Outputs
    box(
        ax, c4, 59, w4, 32, BLUE, "Similarity graph",
        f"each disease → its {graph['k']} highest-\n  scoring diseases\neach edge records\n"
        "  · each profile's contribution\n  · shared phenotypes, genes, ...\n"
        "  · percentile among random pairs\n  · the curated relation, if any",
        footer=f"{graph['edges']:,} edges, {n_diseases:,} diseases",
    )
    support = graph["edge_support"]
    for k, (level, meaning) in enumerate((("curated", "curated relation"), ("plausible", "shared group, gene or drug"), ("novel", "none: a hypothesis"))):
        yy = 70.0 - k * 1.9
        ax.text(c4 + 1.6, yy, f"{support[level]:,}", fontsize=8.4, color=BLUE[0], fontweight="bold", va="center")
        ax.text(c4 + 8.0, yy, f"{level}: {meaning}", fontsize=8.4, color=TEXT, va="center")
    box(
        ax, c4, 24, w4, 29, BLUE, "Placing a new disease",
        "JSON: a name plus any of phenotypes,\n  genes, drugs, text, onset, ...\n"
        "1  encode with the fitted encoders\n2  refit the fusion (< 1 s) with its\n    missing profiles hidden\n"
        f"3  rank all {n_diseases:,} diseases and\n    explain the closest ones\n--add inserts it into the graph",
        footer="CDKL5 example: all four CDKL5\ndisorders in its top 8",
        dashed=True, fill=WHITE, size=8.8,
    )

    # Arrows
    arrow(ax, (c1 + w1, 75), (c2, 75), GRAY[0])
    arrow(ax, (c2 + w2, 75), (c3, 75), BLUE[0])
    arrow(ax, (c3 + w3, 75), (c4, 75), BLUE[0])
    arrow(ax, (c1 + w1, 38.5), (c2, 38.5), ORANGE[0])
    arrow(ax, (c2 + w2, 38.5), (c3, 38.5), ORANGE[0])
    note(ax, c2 + w2 + 3.0, 41.6, "fit", ORANGE[0])
    arrow(ax, (c2 + 17.0, 59), (c2 + 17.0, 53), BLUE[0])
    note(ax, c2 + 18.2, 56.0, "cosines", BLUE[0], ha="left")
    arrow(ax, (c3 + w3, 38.5), (c4, 38.5), BLUE[0], dashed=True)
    note(ax, c3 + w3 + 3.0, 41.6, "refit", BLUE[0])
    arrow(ax, (c4 + 16.5, 53), (c4 + 16.5, 59), BLUE[0], dashed=True)
    note(ax, c4 + 17.7, 56.0, "--add", BLUE[0], ha="left")

    evaluation_strip(ax, metrics, n_diseases)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=160, facecolor=WHITE)
    print(OUTPUT)


if __name__ == "__main__":
    main()
