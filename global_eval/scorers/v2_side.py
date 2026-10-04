"""Defines the evaluation tasks and scores them with v1 and v2 (own process; imports ``v2/``).

Writes to .data/global_eval/:
  tasks.json                    catalogue ids, paper pairs, relation queries
  scores/<task>/<model>.npy     float32 (queries x catalogue) for v1, v1_drugfree, v2, v2_drugfree

Tasks: ``paper_pairs`` (no masking) and ``rel_<relation>`` (the modalities that define the relation are hidden,
exactly as in v2/benchmarks.py).
"""
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(ROOT / "v2")); warnings.filterwarnings("ignore")

import core                                    # global_eval/core.py (a unique module name)
import data_sources                            # v2
from benchmarks import BENCHMARK_SPECS         # v2
from common import assign_split                # v2
from production import load_model              # v2
from scoring import Gallery, V1Baseline        # v2
from paper_pairs_data import load_pairs

QUERIES_PER_RELATION = 400
SEED = 0


def build_tasks(model) -> dict:
    ids = list(model.ids)
    index = set(ids)
    pairs = load_pairs(index)
    for p in pairs:   # which curated relations (v2/v3 training labels) already cover the pair
        key = tuple(sorted((p["a"], p["b"])))
        p["curated"] = sorted(k for k, e in model.relation_evidence.items() if key in e or key[::-1] in e)
    rng = np.random.default_rng(SEED)
    # Curated relations: the pairs v2's production model was fitted to, grouped by query disease.
    relations = {}
    for kind, evidence in model.relation_evidence.items():
        neighbours: dict[str, set] = {}
        for a, b in evidence:
            if a in index and b in index and a != b:
                neighbours.setdefault(a, set()).add(b)
                neighbours.setdefault(b, set()).add(a)
        eligible = sorted(q for q, n in neighbours.items() if 0 < len(n) < len(ids) - 1)
        chosen = sorted(rng.choice(eligible, min(QUERIES_PER_RELATION, len(eligible)), replace=False).tolist())
        relations[kind] = {
            "n_pairs": len(evidence), "n_eligible_queries": len(eligible),
            "queries": {q: sorted(neighbours[q]) for q in chosen},
        }
    return {
        "catalogue": ids,
        "names": dict(zip(ids, model.names)),
        "split": {i: assign_split(i) for i in ids},
        "paper_pairs": pairs,
        "relations": relations,
    }


def main() -> None:
    model = load_model(); bundle = data_sources.load_bundle()
    ids = list(model.ids)
    tasks = build_tasks(model)
    core.write_json(core.DATA / "tasks.json", tasks)
    print(f"[v2] {len(tasks['paper_pairs'])} paper pairs; relations: "
          + ", ".join(f"{k} {len(v['queries'])}/{v['n_eligible_queries']} queries" for k, v in tasks["relations"].items()))

    engine = model.engine; gallery = Gallery(engine, range(len(ids))); v1 = V1Baseline(bundle.v1_sets, ids)

    def score_task(name, queries, masked, v1_masked):
        out = {m: [] for m in ("v1", "v1_drugfree", "v2", "v2_drugfree")}
        for s in range(0, len(queries), 64):
            chunk = queries[s:s + 64]
            Qm, Qa = engine.rows(chunk)
            S, A = engine.block(Qm, Qa, gallery, masked=masked)
            out["v2"].append(model.fusion.score(S, A))
            S2, A2 = engine.block(Qm, Qa, gallery, masked=tuple(set(masked) | set(core.DRUG_DERIVED)))
            out["v2_drugfree"].append(model.fusion.score(S2, A2))
            out["v1"].append(v1.block(chunk, ids, v1_masked))
            out["v1_drugfree"].append(v1.block(chunk, ids, tuple(set(v1_masked) | {"approved_drugs"})))
        folder = core.DATA / "scores" / name
        folder.mkdir(parents=True, exist_ok=True)
        for m, blocks in out.items():
            np.save(folder / f"{m}.npy", np.concatenate(blocks).astype(np.float32))
        print(f"[v2] scored {name}: {len(queries)} queries")

    pair_queries = sorted({p[k] for p in tasks["paper_pairs"] for k in ("a", "b")})
    score_task("paper_pairs", pair_queries, (), ())
    kind_to_spec = {"orphanet_siblings": "orphanet_siblings", "shared_causal_gene": "shared_causal_gene", "shared_drug": "shared_drug"}
    for kind, spec_name in kind_to_spec.items():
        spec = BENCHMARK_SPECS[spec_name]
        score_task(f"rel_{kind}", list(tasks["relations"][kind]["queries"]), spec.masked, spec.v1_masked)


if __name__ == "__main__":
    main()
