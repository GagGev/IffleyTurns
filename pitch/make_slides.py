"""Draw the two pitch slides to ``pitch/``: the knowledge graph and its evaluation.

Numbers are read from v2's graph and evaluation outputs and from ``global_eval/results/metrics.json``.

Usage:  python pitch/make_slides.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
GRAPH = ROOT / "v2" / ".data" / "graph"
V2_METRICS = ROOT / "v2" / ".data" / "evaluation" / "metrics.json"
GLOBAL_METRICS = ROOT / "global_eval" / "results" / "metrics.json"

TEXT, MUTED, WHITE = "#1f2933", "#52606d", "#ffffff"
GRAY = ("#7b8794", "#f5f7fa")
BLUE = ("#2f6db3", "#eaf2fb")
TEAL = ("#1f7a8c", "#e6f3f6")
ORANGE = ("#c46a1b", "#fdf1e6")
GREEN = ("#2b8a5b", "#e8f6ee")
PURPLE = ("#7b4fb3", "#f3eefb")
BASELINE = "#d9a066"

EVIDENCE = {
    "phenotype": ("Symptoms (HPO)", "symptoms"),
    "gene": ("Genes", "genes"),
    "pathway": ("Pathways", "pathways"),
    "ot_gene": ("Open Targets genes", "Open Targets genes"),
    "drug": ("Drugs", "drugs"),
    "drug_target": ("Drug targets", "drug targets"),
    "ontology": ("Classification", "classification"),
    "name": ("Names", "names"),
    "text": ("Description text", "description"),
    "inheritance": ("Inheritance", "inheritance"),
    "onset": ("Age of onset", "age of onset"),
    "prevalence": ("Prevalence", "prevalence"),
}
EXAMPLE = "Duchenne muscular dystrophy"
NEIGHBOURS = {
    "Becker muscular dystrophy": "Becker muscular dystrophy",
    "Symptomatic form of muscular dystrophy of Duchenne and Becker in female carriers": "Symptomatic female carriers",
    "Alpha-sarcoglycan-related limb-girdle muscular dystrophy R3": "Alpha-sarcoglycan LGMD R3",
    "POMGNT2-related limb-girdle muscular dystrophy R24": "POMGNT2 LGMD R24",
    "Emery-Dreifuss muscular dystrophy": "Emery-Dreifuss dystrophy",
}
SUPPORT_COLORS = {"curated": GREEN[0], "plausible": BLUE[0], "novel": ORANGE[0]}


def read(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"{path} not found. Run v2/run_evaluation.py, v2/build_graph.py and global_eval/run.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def canvas():
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 90)
    ax.axis("off")
    return fig, ax


def panel(ax, x, y, w, h, palette, fill=None, radius=1.4, linewidth=1.6):
    edge, face = palette
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}", linewidth=linewidth,
                                edgecolor=edge, facecolor=fill or face))


def header(ax, title, subtitle):
    ax.text(4, 86.5, title, fontsize=26, fontweight="bold", color=TEXT, va="top")
    ax.text(4, 80.2, subtitle, fontsize=14, color=MUTED, va="top")


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=120, facecolor=WHITE)
    plt.close(fig)
    print(path)


# --------------------------------------------------------------------------- slide 1


def neighbourhood():
    edges = pd.read_csv(GRAPH / "edges.csv")
    rows = []
    for full, short in NEIGHBOURS.items():
        match = edges[((edges.source_name == EXAMPLE) & (edges.target_name == full))
                      | ((edges.target_name == EXAMPLE) & (edges.source_name == full))]
        if match.empty:
            raise SystemExit(f"no {EXAMPLE} -- {full} edge in {GRAPH / 'edges.csv'}")
        edge = match.iloc[0]
        reasons = [part.split(":")[0].strip() for part in str(edge.top_modalities).split(";")][:3]
        rows.append((short, edge.score, edge.support, ", ".join(EVIDENCE[m][1] for m in reasons)))
    return rows


def slide_knowledge_graph():
    graph = read(GRAPH / "graph_summary.json")
    splits = read(V2_METRICS)["data"]["split_sizes"]
    fig, ax = canvas()
    header(ax, "A rare-disease knowledge graph where every link has a reason",
           "Built from Orphanet, HPO, Mondo, Reactome, Open Targets, ChEMBL, ClinGen, ClinVar and FDA/EMA orphan designations")

    stats = ((f"{graph['nodes']:,}", "Orphanet rare diseases"),
             (f"{graph['edges']:,}", "links, each stored with\nthe evidence behind it"),
             (str(len(EVIDENCE)), "kinds of evidence\ndescribe each disease"))
    for k, (number, caption) in enumerate(stats):
        y = 56 - k * 21
        panel(ax, 4, y, 36, 18, BLUE)
        ax.text(7, y + 12.3, number, fontsize=38, fontweight="bold", color=BLUE[0], va="center")
        ax.text(7, y + 4.6, caption, fontsize=13, color=TEXT, va="center", linespacing=1.3)

    ax.text(46, 76, "From evidence to one score", fontsize=15, fontweight="bold", color=TEXT, va="center")
    for k, (label, _) in enumerate(EVIDENCE.values()):
        x, y = 46 + (k % 2) * 27.5, 69.5 - (k // 2) * 4.4
        ax.add_patch(FancyBboxPatch((x, y), 26.5, 3.4, boxstyle="round,pad=0,rounding_size=0.8", linewidth=1.2,
                                    edgecolor=BLUE[0], facecolor=WHITE))
        ax.text(x + 13.25, y + 1.7, label, fontsize=12, color=TEXT, ha="center", va="center")
    ax.add_patch(FancyArrowPatch((73, 46.9), (73, 43.0), arrowstyle="-|>", mutation_scale=22, linewidth=2.2, color=BLUE[0]))
    panel(ax, 46, 29, 54, 13.5, BLUE)
    ax.text(48.5, 40.4, "Logistic regression", fontsize=16, fontweight="bold", color=BLUE[0], va="center")
    ax.text(48.5, 36.9, "compares two diseases on each kind of evidence,\nlearning how much each one counts",
            fontsize=11.5, color=TEXT, va="top", linespacing=1.35)
    ax.text(48.5, 30.9, "→  one score for how related they are", fontsize=12.5, fontweight="bold", color=BLUE[0], va="center")

    ax.text(46, 24.6, "Trained, tuned and tested on separate diseases", fontsize=12.5, fontweight="bold", color=TEXT, va="center")
    total, x = sum(splits.values()), 46.0
    for name, palette in (("train", BLUE), ("validation", ORANGE), ("test", GREEN)):
        n = splits[name]
        w = 54 * n / total
        ax.add_patch(Rectangle((x, 17.0), w, 4.6, linewidth=1.5, edgecolor=WHITE, facecolor=palette[0]))
        ax.text(x + w / 2, 19.3, f"{round(100 * n / total)}%", fontsize=13, fontweight="bold", color=WHITE, ha="center", va="center")
        ax.text(x + w / 2, 15.8, f"{name}\n{n:,}", fontsize=10.5, color=palette[0], ha="center", va="top", linespacing=1.25)
        x += w

    ax.text(104, 76, f"Example: {EXAMPLE}", fontsize=15, fontweight="bold", color=TEXT, va="center")
    centre = (110.0, 46.0)
    rows = neighbourhood()
    largest = max(score for _, score, _, _ in rows)
    for k, (name, score, support, reasons) in enumerate(rows):
        node = (124.0, 68.0 - k * 10.0)
        ax.plot([centre[0], node[0]], [centre[1], node[1]], color=SUPPORT_COLORS[support], linewidth=1.6 + 5.0 * score / largest,
                alpha=0.85, solid_capstyle="round", zorder=1)
        ax.add_patch(Circle(node, 1.5, facecolor=WHITE, edgecolor=SUPPORT_COLORS[support], linewidth=2.2, zorder=2))
        ax.text(127.0, node[1] + 0.5, name, fontsize=12, fontweight="bold", color=TEXT, va="bottom")
        ax.text(127.0, node[1] - 0.4, reasons, fontsize=10.5, color=MUTED, va="top")
    ax.add_patch(Circle(centre, 2.8, facecolor="#c0392b", edgecolor=WHITE, linewidth=2.0, zorder=3))
    ax.text(centre[0], centre[1] - 4.2, "Duchenne\nmuscular dystrophy", fontsize=12, fontweight="bold", color=TEXT,
            ha="center", va="top", linespacing=1.25, zorder=4,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": WHITE, "edgecolor": "none", "alpha": 0.92})
    for k, (support, label) in enumerate((("curated", "link backed by a curated relation"),
                                          ("plausible", "plausible: shared group, gene or drug"))):
        y = 19.5 - k * 3.2
        ax.plot([104, 108], [y, y], color=SUPPORT_COLORS[support], linewidth=4, solid_capstyle="round")
        ax.text(109.5, y, label, fontsize=11, color=TEXT, va="center")
    ax.text(104, 12.5, "Grey text: the evidence that contributed most to each link", fontsize=10.5, color=MUTED, va="center", style="italic")

    support = graph["edge_support"]
    share = {level: round(100 * n / graph["edges"]) for level, n in support.items()}
    ax.text(4, 6.0, f"Each disease links to its {graph['k']} highest-scoring diseases. {share['curated']}% of links match a curated "
            f"relation, {share['plausible']}% are plausible, and {share['novel']}% are new hypotheses for experts to review.",
            fontsize=12.5, color=TEXT, va="center")
    save(fig, "slide_1_knowledge_graph.png")


# --------------------------------------------------------------------------- slide 2


def bar_rows(ax, x, y, ours, baseline, palette):
    for k, (label, value, color) in enumerate((("our model", ours, palette[0]), ("baseline", baseline, BASELINE))):
        yy = y - k * 3.0
        ax.text(x, yy, label, fontsize=10.5, color=TEXT, va="center")
        ax.add_patch(Rectangle((x + 9.5, yy - 0.9), 20.0 * value, 1.8, linewidth=0, facecolor=color))
        ax.text(x + 9.5 + 20.0 * value + 0.7, yy, f"{100 * value:.0f}%", fontsize=10.5, fontweight="bold", color=TEXT, va="center")


def slide_evaluation():
    m = read(GLOBAL_METRICS)
    paper = m["paper_pairs"]
    hits = paper["disease_level_hits"]
    held_out = read(V2_METRICS)["test"]
    siblings = held_out["orphanet_siblings"]["all"]
    genes = held_out["shared_causal_gene"]["all"]["models"]
    forward = m["forward_time"]["folds"]["test"]
    future = forward["metrics"]["full"]
    symptoms = m["symptom_retrieval"]
    catalogue = 7493

    fig, ax = canvas()
    header(ax, "Four ways to check that the links are real",
           f"Each test gives the model one disease and asks it to rank the other {catalogue - 1:,}. Do the true partners come out on top?")

    tiles = [
        (TEAL, "Paper-described pairs",
         f"{paper['n_pairs']} disease pairs that published\npapers call related or similar",
         hits["all"]["v2"]["top10"], "of these diseases have a paper\npartner in their top 10",
         f"no curated link between them: {100 * hits['no_curated_relation']['v2']['top10']:.0f}%\n"
         f"(baseline {100 * hits['no_curated_relation']['v1']['top10']:.0f}%)",
         hits["all"]["v1"]["top10"]),
        (ORANGE, "Curated relations, hidden",
         "Orphanet sibling diseases, with\nclassification and names hidden",
         siblings["models"]["logistic_fusion"]["hits@10"]["mean"],
         f"of {siblings['n_queries']:,} unseen test diseases have\na true sibling in their top 10",
         f"shared causal gene, gene data hidden:\n"
         f"{100 * genes['logistic_fusion']['hits@10']['mean']:.0f}% "
         f"(baseline {100 * genes['v1_weighted_jaccard']['hits@10']['mean']:.0f}%)",
         siblings["models"]["v1_weighted_jaccard"]["hits@10"]["mean"]),
        (GREEN, "Future orphan-drug links",
         f"trained on data before 2018, then\npredicts {forward['info']['new_relations']:,} later links",
         future["v2_retrained"]["hits@10"]["mean"], "of diseases get a future drug\npartner in their top 10",
         f"random ranking: {100 * future['random']['hits@10']['mean']:.0f}%",
         future["v1_weighted_jaccard"]["hits@10"]["mean"]),
        (PURPLE, "Symptom-only search",
         "the model gets just 5 of a\ndisease's symptoms, nothing else",
         symptoms["k5"]["v2"]["top10"], "find the right disease\nin the top 10",
         f"with only 3 symptoms: {100 * symptoms['k3']['v2']['top10']:.0f}%\n(baseline {100 * symptoms['k3']['v1']['top10']:.0f}%)",
         symptoms["k5"]["v1"]["top10"]),
    ]
    w, gap, y0, h = 36.25, 2.0, 27.5, 48.0
    for k, (palette, name, what, value, caption, extra, baseline) in enumerate(tiles):
        x = 4 + k * (w + gap)
        panel(ax, x, y0, w, h, palette)
        ax.text(x + 2, y0 + h - 2.4, f"{k + 1}  {name}", fontsize=15, fontweight="bold", color=palette[0], va="top")
        ax.text(x + 2, y0 + h - 7.6, what, fontsize=12, color=TEXT, va="top", linespacing=1.3)
        ax.text(x + w / 2, y0 + 26.0, f"{100 * value:.0f}%", fontsize=50, fontweight="bold", color=palette[0], ha="center", va="center")
        ax.text(x + w / 2, y0 + 19.6, caption, fontsize=12.5, color=TEXT, ha="center", va="top", linespacing=1.3)
        ax.text(x + w / 2, y0 + 12.2, extra, fontsize=10.5, color=MUTED, ha="center", va="top", linespacing=1.3)
        bar_rows(ax, x + 2, y0 + 5.2, value, baseline, palette)

    ax.text(4, 23.6, "How we measure", fontsize=14, fontweight="bold", color=TEXT, va="center")
    metrics = (("MAP", "are all the true partners\nnear the top?"),
               ("MRR", "how high is the first\ntrue partner?"),
               ("Hits@k", "is a true partner\nin the top k?"),
               ("Median rank", "the typical position\nof the partner"),
               ("Partner percentile", "share of diseases ranked\nbelow the partner"))
    mw = (152 - 4 * 1.5) / 5
    for k, (name, meaning) in enumerate(metrics):
        x = 4 + k * (mw + 1.5)
        panel(ax, x, 7.5, mw, 13.0, GRAY, fill=WHITE, radius=1.0, linewidth=1.2)
        ax.text(x + 1.8, 17.6, name, fontsize=15, fontweight="bold", color=BLUE[0], va="center")
        ax.text(x + 1.8, 14.4, meaning, fontsize=11.5, color=TEXT, va="top", linespacing=1.3)
    ax.text(4, 3.6, "Baseline: v1, a weighted overlap of shared annotations. Hit rates are per query disease. Test 2 uses v2's "
            "held-out test diseases; the others are from global_eval/results/report.md (with 95% intervals).",
            fontsize=10.5, color=MUTED, va="center")
    save(fig, "slide_2_evaluation.png")


if __name__ == "__main__":
    slide_knowledge_graph()
    slide_evaluation()
