"""Score global-evaluation tasks with the standalone v4 embedding model."""

import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROOT / "v4"))
warnings.filterwarnings("ignore")

import core
from model import load_model

SPECS = {
    "orphanet_siblings": ("ontology", "name"),
    "shared_causal_gene": ("gene", "pathway"),
    "shared_drug": (),
}


def main() -> None:
    tasks = core.read_json(core.DATA / "tasks.json")
    model = load_model()
    model_index = {disease: index for index, disease in enumerate(model.ids)}
    missing = sorted(set(tasks["catalogue"]) - set(model_index))
    if missing:
        raise RuntimeError(f"v4 model is missing {len(missing)} global-eval diseases: {missing[:5]}")
    columns = np.array([model_index[disease] for disease in tasks["catalogue"]], dtype=np.int64)

    def run(name: str, queries: list[str], masked: tuple[str, ...]) -> None:
        blocks = []
        for start in range(0, len(queries), 128):
            scores = model.score_queries(queries[start : start + 128], masked=masked)
            blocks.append(scores[:, columns])
        output = np.concatenate(blocks).astype(np.float32)
        if output.shape != (len(queries), len(tasks["catalogue"])):
            raise RuntimeError(f"Bad v4 score shape for {name}: {output.shape}")
        folder = core.DATA / "scores" / name
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "v4_embedding.npy", output)
        print(f"[v4] scored {name}: {len(queries)} queries", flush=True)

    pair_queries = sorted({pair[key] for pair in tasks["paper_pairs"] for key in ("a", "b")})
    run("paper_pairs", pair_queries, ())
    for kind, masked in SPECS.items():
        run(f"rel_{kind}", list(tasks["relations"][kind]["queries"]), masked)


if __name__ == "__main__":
    main()
