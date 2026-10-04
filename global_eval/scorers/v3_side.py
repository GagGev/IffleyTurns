"""Scores the tasks with the v3 production model (own process; imports ``v3/``).  Needs PyTorch.

Reads .data/global_eval/tasks.json (written by v2_side.py) and writes scores/<task>/v3_static.npy and, for the
unmasked paper-pair task, v3_forecast.npy.  For the relation tasks the same modalities are hidden as for v2:
from the similarity matrices and from the neural embeddings of both diseases.
"""
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(ROOT / "v3")); warnings.filterwarnings("ignore")

import core
from production import load_model     # v3

SPECS = {   # hidden modalities per relation (v2/benchmarks.py), restricted to v3's drug-free modality set
    "orphanet_siblings": ("ontology", "name"),
    "shared_causal_gene": ("gene", "pathway"),
    "shared_drug": (),
}


def main() -> None:
    tasks = core.read_json(core.DATA / "tasks.json")
    model = load_model(); w = model.world; sim = model._sim
    cols = np.array([w.index[i] for i in tasks["catalogue"]])
    hybrid = model.static

    def score(rows, masked, forecast):
        S, A = sim.block(rows, masked=masked)
        if masked:
            z = hybrid.neural.embeddings(w.matrices, w.available, masked=masked)
            logits = []
            for member, zm in zip(hybrid.neural.models, z):
                n = S.shape[1]
                logits.append(member.pair_logits(np.repeat(zm[rows], n, axis=0), np.tile(zm, (len(rows), 1)),
                                                 S.reshape(-1, S.shape[-1]), A.reshape(-1, A.shape[-1])).reshape(len(rows), n))
            neural = np.mean(logits, axis=0)
        else:
            neural = hybrid.neural.block(rows, S, A)
        static = hybrid.combine(neural, S, A)["static_logit"]
        out = {"v3_static": static[:, cols]}
        if forecast:
            out["v3_forecast"] = model.score_nodes(rows)["forecast"][:, cols]
        return out

    def run(name, queries, masked, forecast):
        rows = np.array([w.index[q] for q in queries])
        blocks = {}
        for s in range(0, len(rows), 32):
            for key, value in score(rows[s:s + 32], masked, forecast).items():
                blocks.setdefault(key, []).append(value)
        folder = core.DATA / "scores" / name
        folder.mkdir(parents=True, exist_ok=True)
        for key, parts in blocks.items():
            np.save(folder / f"{key}.npy", np.concatenate(parts).astype(np.float32))
        print(f"[v3] scored {name}: {len(queries)} queries", flush=True)

    pair_queries = sorted({p[k] for p in tasks["paper_pairs"] for k in ("a", "b")})
    run("paper_pairs", pair_queries, (), True)
    for kind, masked in SPECS.items():
        run(f"rel_{kind}", list(tasks["relations"][kind]["queries"]), masked, False)


if __name__ == "__main__":
    main()
