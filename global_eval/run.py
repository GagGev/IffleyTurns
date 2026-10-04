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
import forward_time, paper_pairs, relations, symptom_retrieval

BENCHMARKS = {m.NAME: m for m in (paper_pairs, relations, symptom_retrieval, forward_time)}
SCORE_TASKS = ("paper_pairs", "symptoms", "rel_orphanet_siblings", "rel_shared_causal_gene", "rel_shared_drug")
RELATION_TASKS = ("paper_pairs", "rel_orphanet_siblings", "rel_shared_causal_gene", "rel_shared_drug")
SCORER_MODELS = {
    "v2": ("v1", "v1_drugfree", "v2", "v2_drugfree"),
    "v3": ("v3_static", "v3_forecast"),
    "v4": ("v4_embedding",),
}
SCORER_SCRIPTS = {"v2": "v2_side.py", "v3": "v3_side.py", "v4": "v4_side.py"}


def scorer_versions(model_ids) -> list[str]:
    selected = set(model_ids)
    return [
        version
        for version in ("v2", "v3", "v4")
        if selected.intersection(SCORER_MODELS[version])
    ]


def _model_tasks(model: str):
    if model == "v3_forecast":
        return ("paper_pairs",)
    if model in ("v1_drugfree", "v2_drugfree", "v4_embedding"):
        return RELATION_TASKS
    return SCORE_TASKS


def _expected_outputs(version: str, model_ids) -> list[Path]:
    selected = set(model_ids).intersection(SCORER_MODELS[version])
    return [
        core.DATA / "scores" / task / f"{model}.npy"
        for model in selected
        for task in _model_tasks(model)
    ]


def score(rescore: bool, model_ids) -> None:
    versions = scorer_versions(model_ids)
    if not (core.DATA / "tasks.json").is_file() and "v2" not in versions:
        versions.insert(0, "v2")  # v2_side owns the common task definition
    for version in versions:
        expected = _expected_outputs(version, model_ids)
        prerequisite = version == "v2" and not (core.DATA / "tasks.json").is_file()
        done = (core.DATA / "tasks.json").is_file() and expected and all(path.is_file() for path in expected)
        if done and not rescore:
            print(f"{version} score matrices found; use --rescore to recompute")
            continue
        if not expected and not prerequisite:
            continue
        script = SCORER_SCRIPTS[version]
        print(f"== {script}", flush=True)
        subprocess.run([sys.executable, str(HERE / "scorers" / script)], check=True)


def headline(metrics: dict, model_ids) -> str:
    """One table across the benchmarks: the like-for-like (drug-free) rows of each version."""
    pp = metrics.get("paper_pairs", {}).get("strata", {})
    rel = metrics.get("relations", {})
    sym = metrics.get("symptom_retrieval", {})
    fwd = metrics.get("forward_time", {}).get("folds", {}).get("test", {}).get("metrics", {}).get("full", {})
    # (label, paper-pair model id, relation model id, forward-time model id)
    selected = set(model_ids)
    definitions = [
        ("v1 weighted Jaccard", "v1", "v1", "v1", "v1_weighted_jaccard"),
        ("v2 (drug-free view; retrained for forward time)", "v2_drugfree", "v2_drugfree", "v2", "v2_retrained"),
        ("v3 static similarity", "v3_static", "v3_static", "v3_static", "v3_static"),
        ("v3 forecast / stacker", "v3_forecast", None, None, "v3_stacker"),
        ("v4 embedding cosine", "v4_embedding", "v4_embedding", None, None),
    ]
    rows = [
        row for row in definitions
        if row[1] in selected or row[2] in selected or row[3] in selected
    ]
    out = []
    for label, pm, rm, sm, fm in rows:
        cells = [label]
        dl = metrics.get("paper_pairs", {}).get("disease_level_hits", {})
        for stratum in ("all", "no_curated_relation"):
            e = dl.get(stratum, {}).get(pm)
            cells.append(f"{100 * e['top10']:.0f}%" if e else "–")
        for kind in ("orphanet_siblings", "shared_causal_gene", "shared_drug"):
            e = rel.get(kind, {}).get("models", {}).get(rm) if rm else None
            cells.append(f"{100 * e['hits@10']['mean']:.0f}%" if e else "–")
        e = sym.get("k5", {}).get(sm) if sm else None
        cells.append(f"{100 * e['top10']:.0f}%" if e else "–")
        e = fwd.get(fm)
        cells.append(f"{100 * e['hits@10']['mean']:.0f}%" if e else "–")
        out.append(cells)
    header = ["Model", "Paper pairs, all", "Paper pairs, no curated relation", "Siblings*", "Shared gene*", "Shared drug*",
              "5 symptoms only**", "Forward in time"]
    relation_note = (
        "\\* v2's curated-relation rows are in-sample. v4's all-query sibling/gene rows mix splits; "
        "its detailed test-split values are held out."
        if "v4_embedding" in selected
        else "\\* In-sample for v2 and v3 (see the relations benchmark)."
    )
    return (
        "## Headline: share of diseases with a valid partner in their top 10\n\n"
        + core.md_table(header, out)
        + "\n\nEach cell is the percentage of query diseases with at least one valid partner among their 10 highest-scored diseases"
        " (paper-stated partner, curated relation, the disease itself for a 5-symptom query, or a relation formed after the 2018 cutoff). "
        "\\*\\* Query is 5 of the disease's own phenotypes and nothing else; v2 there uses all 12 modalities. "
        + relation_note
        + " Forward in time is the test fold over 9,525 nodes; v2 there is the v2 architecture retrained"
        " on pre-cutoff relations. MAP, MRR and AUROC are in the sections below.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rescore", action="store_true")
    parser.add_argument("--only", nargs="+", choices=list(BENCHMARKS), default=list(BENCHMARKS))
    parser.add_argument("--models", nargs="+", choices=list(core.MODELS), default=list(core.MODELS))
    parser.add_argument("--output-dir", type=Path, default=core.RESULTS)
    args = parser.parse_args()

    if any(b != "forward_time" for b in args.only):
        score(args.rescore, args.models)
    core.RESULTS = args.output_dir.resolve()
    tasks = core.read_json(core.DATA / "tasks.json") if (core.DATA / "tasks.json").is_file() else {}

    metrics, sections = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "models": args.models,
    }, []
    for name in args.only:
        module = BENCHMARKS[name]
        print(f"== {name}", flush=True)
        if name in ("paper_pairs", "relations"):
            result, text = module.run(tasks, args.models)
        else:
            result, text = module.run(tasks)
        metrics[name] = result
        sections.append(f"## {module.TITLE}\n\n{module.INTRO}\n\n{text}")
    core.write_json(core.RESULTS / "metrics.json", metrics)
    header_name = "report_header_v4.md" if "v4_embedding" in args.models else "report_header.md"
    header = (HERE / header_name).read_text(encoding="utf-8")
    (core.RESULTS / "report.md").write_text(
        header + "\n\n" + headline(metrics, args.models) + "\n" + "\n\n".join(sections) + "\n",
        encoding="utf-8",
    )
    print("wrote", core.RESULTS / "report.md")


if __name__ == "__main__":
    main()
