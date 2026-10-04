"""Global evaluation of v1, v2 and v3 on common benchmarks.

    python global_eval/run.py                  # score (if needed) and evaluate everything
    python global_eval/run.py --rescore        # recompute the score matrices
    python global_eval/run.py --only paper_pairs relations forward_time

Scoring runs v1/v2 and v3 in separate processes (their modules share names), using this interpreter, then the
benchmarks read the matrices.  v3 needs PyTorch.  Outputs: global_eval/results/{report.md, metrics.json, ...}.
"""
import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "benchmarks"))

import core
import forward_time, paper_pairs, relations

BENCHMARKS = {m.NAME: m for m in (paper_pairs, relations, forward_time)}


def score(rescore: bool) -> None:
    done = (core.DATA / "tasks.json").is_file() and all(
        (core.DATA / "scores" / t / f).is_file()
        for t in ("paper_pairs", "rel_orphanet_siblings", "rel_shared_causal_gene", "rel_shared_drug")
        for f in ("v2.npy", "v3_static.npy"))
    if done and not rescore:
        print("score matrices found; use --rescore to recompute")
        return
    for script in ("v2_side.py", "v3_side.py"):
        print(f"== {script}", flush=True)
        subprocess.run([sys.executable, str(HERE / "scorers" / script)], check=True)


def headline(metrics: dict) -> str:
    """One table across the benchmarks: the like-for-like (drug-free) rows of each version."""
    pp = metrics.get("paper_pairs", {}).get("strata", {})
    rel = metrics.get("relations", {})
    fwd = metrics.get("forward_time", {}).get("folds", {}).get("test", {}).get("metrics", {}).get("full", {})
    # (label, paper-pair model id, relation model id, forward-time model id)
    rows = [("v1 weighted Jaccard", "v1", "v1", "v1_weighted_jaccard"),
            ("v2 (drug-free view; retrained for forward time)", "v2_drugfree", "v2_drugfree", "v2_retrained"),
            ("v3 static similarity", "v3_static", "v3_static", "v3_static"),
            ("v3 forecast / stacker", "v3_forecast", None, "v3_stacker")]
    out = []
    for label, pm, rm, fm in rows:
        cells = [label]
        dl = metrics.get("paper_pairs", {}).get("disease_level_hits", {})
        for stratum in ("all", "no_curated_relation"):
            e = dl.get(stratum, {}).get(pm)
            cells.append(f"{100 * e['top10']:.0f}%" if e else "–")
        for kind in ("orphanet_siblings", "shared_causal_gene", "shared_drug"):
            e = rel.get(kind, {}).get("models", {}).get(rm) if rm else None
            cells.append(f"{100 * e['hits@10']['mean']:.0f}%" if e else "–")
        e = fwd.get(fm)
        cells.append(f"{100 * e['hits@10']['mean']:.0f}%" if e else "–")
        out.append(cells)
    header = ["Model", "Paper pairs, all", "Paper pairs, no curated relation", "Siblings*", "Shared gene*", "Shared drug*",
              "Forward in time"]
    return ("## Headline: share of diseases with a valid partner in their top 10\n\n" + core.md_table(header, out)
            + "\n\nEach cell is the percentage of query diseases with at least one valid partner among their 10 highest-scored diseases"
              " (paper-stated partner, curated relation, or a relation formed after the 2018 cutoff). \\* In-sample for v2 and v3 (see"
              " the relations benchmark). Forward in time is the test fold over 9,525 nodes; v2 there is the v2 architecture retrained"
              " on pre-cutoff relations. MAP, MRR and AUROC are in the sections below.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rescore", action="store_true")
    parser.add_argument("--only", nargs="+", choices=list(BENCHMARKS), default=list(BENCHMARKS))
    args = parser.parse_args()

    if any(b != "forward_time" for b in args.only):
        score(args.rescore)
    tasks = core.read_json(core.DATA / "tasks.json") if (core.DATA / "tasks.json").is_file() else {}

    metrics, sections = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds")}, []
    for name in args.only:
        module = BENCHMARKS[name]
        print(f"== {name}", flush=True)
        result, text = module.run(tasks)
        metrics[name] = result
        sections.append(f"## {module.TITLE}\n\n{module.INTRO}\n\n{text}")
    core.write_json(core.RESULTS / "metrics.json", metrics)
    header = (HERE / "report_header.md").read_text(encoding="utf-8")
    (core.RESULTS / "report.md").write_text(header + "\n\n" + headline(metrics) + "\n" + "\n\n".join(sections) + "\n", encoding="utf-8")
    print("wrote", core.RESULTS / "report.md")


if __name__ == "__main__":
    main()
