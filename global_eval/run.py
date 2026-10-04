"""Global evaluation of versioned models on common benchmarks.

    python global_eval/run.py                  # score (if needed) and evaluate everything
    python global_eval/run.py --rescore        # recompute the score matrices
    python global_eval/run.py --only paper_pairs relations forward_time
    python global_eval/run.py --models v2 v2_drugfree v4_embedding --only paper_pairs relations

Each requested version is scored in a separate process, then the benchmarks read only score matrices. Selecting
v2 and v4 does not import or execute v3. Outputs default to global_eval/results/ and can be redirected.
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
SCORE_TASKS = ("paper_pairs", "rel_orphanet_siblings", "rel_shared_causal_gene", "rel_shared_drug")
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


def _expected_outputs(version: str, model_ids) -> list[Path]:
    selected = set(model_ids).intersection(SCORER_MODELS[version])
    outputs = []
    for model in selected:
        tasks = ("paper_pairs",) if model == "v3_forecast" else SCORE_TASKS
        outputs.extend(core.DATA / "scores" / task / f"{model}.npy" for task in tasks)
    return outputs


def score(rescore: bool, model_ids) -> None:
    versions = scorer_versions(model_ids)
    tasks_exist = (core.DATA / "tasks.json").is_file()
    # v2_side owns the shared task definition. Run it as a prerequisite only
    # when another version was requested before tasks.json exists.
    if not tasks_exist and "v2" not in versions:
        versions.insert(0, "v2")
    for version in versions:
        expected = _expected_outputs(version, model_ids)
        prerequisite = version == "v2" and not (core.DATA / "tasks.json").is_file()
        done = (core.DATA / "tasks.json").is_file() and expected and all(path.is_file() for path in expected)
        if done and not rescore:
            print(f"{version} score matrices found; use --rescore to recompute")
            continue
        # Do not rerun an unselected v2 prerequisite merely because --rescore
        # was supplied when tasks already exist.
        if not expected and not prerequisite:
            continue
        script = SCORER_SCRIPTS[version]
        print(f"== {script}", flush=True)
        subprocess.run([sys.executable, str(HERE / "scorers" / script)], check=True)


def headline(metrics: dict, model_ids) -> str:
    """One table across the benchmarks: the like-for-like (drug-free) rows of each version."""
    pp = metrics.get("paper_pairs", {}).get("strata", {})
    rel = metrics.get("relations", {})
    fwd = metrics.get("forward_time", {}).get("folds", {}).get("test", {}).get("metrics", {}).get("full", {})
    # (label, paper-pair model id, relation model id, forward-time model id)
    selected = set(model_ids)
    definitions = [
        ("v1 weighted Jaccard", "v1", "v1", "v1_weighted_jaccard"),
        ("v2 (drug-free view; retrained for forward time)", "v2_drugfree", "v2_drugfree", "v2_retrained"),
        ("v3 static similarity", "v3_static", "v3_static", "v3_static"),
        ("v3 forecast / stacker", "v3_forecast", None, "v3_stacker"),
        ("v4 embedding cosine", "v4_embedding", "v4_embedding", None),
    ]
    rows = []
    for definition in definitions:
        _, paper_model, relation_model, _ = definition
        if paper_model in selected or relation_model in selected:
            rows.append(definition)
    out = []
    for label, pm, rm, fm in rows:
        cells = [label]
        for stratum in ("all", "no_curated_relation"):
            e = pp.get(stratum, {}).get("models", {}).get(pm)
            cells.append(f"{e['hits@10']:.2f} / {e['mean_percentile']:.3f}" if e else "–")
        for kind in ("orphanet_siblings", "shared_causal_gene", "shared_drug"):
            e = rel.get(kind, {}).get("models", {}).get(rm) if rm else None
            cells.append(f"{e['map']['mean']:.3f}" if e else "–")
        e = fwd.get(fm)
        cells.append(f"{e['map']['mean']:.3f}" if e else "–")
        out.append(cells)
    header = ["Model", "Paper pairs, all: Hits@10 / mean pctl", "Paper pairs, no curated relation", "Siblings MAP*",
              "Shared gene MAP*", "Shared drug MAP*", "Forward-time MAP"]
    if "v4_embedding" in selected:
        note = (
            "\\* v2's curated-relation rows are in-sample. v4's all-query sibling/gene rows mix splits; "
            "its test-split values in the detailed tables are held out. Paper-pair columns are Hits@10 and "
            "mean partner percentile among 7,493 diseases."
        )
    else:
        note = (
            "\\* In-sample for v2 and v3 (see the relations benchmark). Forward-time MAP is the test fold, "
            "all 9,525 nodes as candidates; v2 there is the v2 architecture retrained on pre-cutoff relations. "
            "Paper-pair columns are Hits@10 and mean partner percentile among 7,493 diseases."
        )
    return "## Headline\n\n" + core.md_table(header, out) + "\n\n" + note + "\n"


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
