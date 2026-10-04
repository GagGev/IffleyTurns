"""Draw the v3 model schematic to ``v3/docs/model_schematic.png``.

Usage:  python v3/make_schematic.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUTPUT = Path(__file__).resolve().parent / "docs" / "model_schematic.png"

TEXT, MUTED = "#1f2933", "#52606d"
GRAY = ("#7b8794", "#f5f7fa")
BLUE = ("#2f6db3", "#eaf2fb")
ORANGE = ("#c46a1b", "#fdf1e6")
GREEN = ("#2b8a5b", "#e8f6ee")
WHITE = "#ffffff"


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
        ax.text(x + 1.6, y + 1.5, footer, fontsize=size, fontweight="bold", color=edge, va="bottom")


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


def chips(ax, x, y, names, palette, per_row=3, w=9.6, h=3.0, gap=0.8):
    edge, _ = palette
    for k, name in enumerate(names):
        cx = x + (k % per_row) * (w + gap)
        cy = y - (k // per_row) * (h + gap)
        ax.add_patch(
            FancyBboxPatch((cx, cy), w, h, boxstyle="round,pad=0,rounding_size=0.8", linewidth=1.0, edgecolor=edge, facecolor=WHITE)
        )
        ax.text(cx + w / 2, cy + h / 2, name, fontsize=8.4, color=TEXT, ha="center", va="center")


def timeline(ax, x0, x1, y):
    def to_x(year: float) -> float:
        return x0 + (year - 1983) / (2027 - 1983) * (x1 - x0)

    ax.text(2, y + 9.0, "Time-honest training and testing", fontsize=11, fontweight="bold", color=TEXT, va="center")
    ax.text(
        2, y + 2.6,
        "Every feature and label comes from a snapshot\nof what was known at the cutoff. The released\n"
        "model is refit on everything up to Sep 2026.",
        fontsize=8.6, color=MUTED, va="center", linespacing=1.4,
    )
    rows = [
        ("validation (model choices)", 2014.0, 2018.0, "532 new relations"),
        ("test (reported results)", 2018.0, 2026.75, "1,123 new relations"),
    ]
    for k, (name, cutoff, end, label) in enumerate(rows):
        yy = y + 6.5 - k * 5.2
        ax.add_patch(FancyBboxPatch((to_x(1983), yy), to_x(cutoff) - to_x(1983), 3.2, boxstyle="square,pad=0", linewidth=0, facecolor=ORANGE[1]))
        ax.add_patch(FancyBboxPatch((to_x(cutoff), yy), to_x(end) - to_x(cutoff), 3.2, boxstyle="square,pad=0", linewidth=0, facecolor=GREEN[0], alpha=0.85))
        ax.text(to_x(1984), yy + 1.6, f"{name}: train on everything known before {int(cutoff)}", fontsize=8.4, color=ORANGE[0], va="center")
        ax.text(to_x(end) + 0.8, yy + 1.6, f"predict {label}", fontsize=8.4, color=GREEN[0], va="center")
    for year in (1983, 1990, 2000, 2010, 2014, 2018, 2026):
        ax.plot([to_x(year)] * 2, [y - 1.2, y - 0.4], color=MUTED, linewidth=0.8)
        ax.text(to_x(year), y - 1.8, str(year), fontsize=8.0, color=MUTED, ha="center", va="top")
    ax.plot([to_x(1983), to_x(2027)], [y - 0.4, y - 0.4], color=MUTED, linewidth=0.8)


def main() -> None:
    fig = plt.figure(figsize=(16, 10.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 106)
    ax.axis("off")

    ax.text(2, 103.2, "v3: how the rare-disease similarity graph and forecast are built", fontsize=17, fontweight="bold", color=TEXT, va="top")
    ax.text(
        2, 98.6,
        "Two diseases count as related when one drug holds separate FDA or EMA orphan designations for both, "
        "an independent regulator's judgement that they share a treatable mechanism.",
        fontsize=10, color=MUTED, va="top",
    )
    c1, c2, c3, c4 = 2, 39, 82, 127
    for x, label in ((c1, "1  Inputs"), (c2, "2  Prepare"), (c3, "3  Models"), (c4, "4  Outputs")):
        ax.text(x, 93.0, label, fontsize=11.5, fontweight="bold", color=MUTED, va="center")

    # 1 Inputs
    box(
        ax, c1, 60, 32, 31, GRAY, "Disease knowledge",
        "Orphanet: classification, names,\n  clinical descriptions, epidemiology\n"
        "HPO phenotypes, Mondo ontology\nGenes: Orphanet, HPO, ClinGen, ClinVar\n"
        "Reactome pathways of those genes\nNo drug information of any kind",
        footer="9,525 nodes: 7,493 diseases\n+ 2,032 Orphanet groups",
    )
    box(
        ax, c1, 45, 32, 11.5, GRAY, "New disease (optional)",
        "phenotypes, genes, text, ... (+ any drugs)\nruns through the same path",
        dashed=True, fill=WHITE,
    )
    box(
        ax, c1, 22, 32, 19, ORANGE, "Regulatory decisions",
        "FDA orphan designations (since 1983)\nEMA orphan designations (since 2000)\n"
        "each: drug, indication text, date,\n  later approval",
        footer="11,124 designation records",
    )

    # 2 Prepare
    box(ax, c2, 60, 34, 31, BLUE, "Nine disease profiles", fill=GRAY[1])
    chips(
        ax, c2 + 1.6, 82.4,
        ("phenotype", "gene", "pathway", "ontology", "name", "text", "inheritance", "onset", "prevalence"),
        BLUE,
    )
    ax.text(
        c2 + 1.6, 72.0,
        "each a weighted sparse vector\n\nFor any pair of diseases:\n9 cosine similarities + 9 flags for\nwhich profiles both diseases have",
        fontsize=9.2, color=TEXT, va="top", linespacing=1.5,
    )
    box(
        ax, c2, 22, 34, 31, ORANGE, "Ground-truth builder",
        "1  Drug name → one molecule (ChEMBL)\n2  Indication text → Orphanet code\n     (75% of designations mapped)\n"
        "3  Same drug, two separate designations\n     → the two diseases are related,\n     dated by the later designation\n"
        "Skipped: parent–subtype pairs and\n  drugs designated for >12 diseases",
        footer="2,473 dated relations",
    )

    # 3 Models
    box(ax, c3, 52, 40, 39, BLUE, "Similarity model (no drug data)")
    box(
        ax, c3 + 1.6, 78.5, 36.8, 8.2, BLUE, None,
        "Linear fusion\nnon-negative logistic regression on the 9 cosines",
        fill=WHITE, size=8.8,
    )
    ax.text(c3 + 20, 76.3, "+   standardized sum   =   similarity score", fontsize=8.8, color=BLUE[0], ha="center", va="center", fontweight="bold")
    box(
        ax, c3 + 1.6, 53.6, 36.8, 20.6, BLUE, None,
        "Neural network (3 seeds, GPU)\n"
        "encoder: each profile → its own layer →\n  attention pooling → 128-number embedding\n"
        "pair head: MLP on both embeddings + cosines",
        fill=WHITE, size=8.8,
    )
    ax.text(
        c3 + 3.2, 55.0,
        "Encoder learns from Orphanet siblings and\nshared genes; only the pair head learns\nfrom the regulatory relations.",
        fontsize=8.4, color=MUTED, va="bottom", linespacing=1.35,
    )
    box(
        ax, c3, 22, 40, 24, GREEN, "Forecast model",
        "Gradient-boosted trees over\n· the similarity scores and cosines\n"
        "· drug mechanism: targets and pathways\n   of drugs already designated\n"
        "· known relations: common neighbours\n· node: group size, oncology, category",
        footer="trained at past dates 2-8 years before the cutoff",
    )

    # 4 Outputs
    box(
        ax, c4, 60, 31, 31, BLUE, "Similarity graph",
        "each disease → its 10 most\n  similar diseases (never its own\n  parent groups or subtypes)\n"
        "each edge explains itself:\n  shared phenotypes and genes,\n  drugs already shared, forecast",
        footer="67,892 edges, 9,525 nodes",
    )
    box(
        ax, c4, 22, 31, 31, GREEN, "Forecast ranking",
        "pairs most likely to share a future\n  orphan designation\n\n"
        "Test (trained before 2018,\n  predicting 2018 to Sep 2026):\n"
        "· of its top 100 pairs, 19% became\n  related (45× the base rate)\n· mean average precision:",
        footer="0.103  (v2 0.060, random 0.001)",
    )

    # Arrows
    arrow(ax, (c1 + 32, 76), (c2, 76), GRAY[0])
    arrow(ax, (c1 + 32, 50.5), (c2 + 5, 60), GRAY[0], dashed=True, rad=0.25)
    arrow(ax, (c1 + 32, 32), (c2, 32), ORANGE[0])
    arrow(ax, (c2 + 34, 76), (c3, 76), BLUE[0])
    arrow(ax, (c3 + 40, 76), (c4, 76), BLUE[0])
    arrow(ax, (c3 + 30, 52), (c3 + 30, 46), BLUE[0])
    note(ax, c3 + 31, 49, "similarity", BLUE[0], ha="left")
    arrow(ax, (c2 + 34, 47), (c3, 57), ORANGE[0], dashed=True, rad=-0.2)
    note(ax, c2 + 38.5, 44.2, "training\nlabels", ORANGE[0])
    arrow(ax, (c2 + 34, 30), (c3, 30), ORANGE[0])
    note(ax, c2 + 38.5, 26.5, "snapshot\nat date T", ORANGE[0])
    arrow(ax, (c3 + 40, 34), (c4, 34), GREEN[0])

    timeline(ax, 40, 140, 6.5)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=160, facecolor=WHITE)
    print(OUTPUT)


if __name__ == "__main__":
    main()
