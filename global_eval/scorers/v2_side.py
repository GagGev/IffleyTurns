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


SYMPTOM_DISEASES = 500
SYMPTOM_VARIANTS = [("k3", 3, 0), ("k5", 5, 0), ("k10", 10, 0), ("k5_noise1", 5, 1)]   # (name, true terms, unrelated terms)


def symptom_queries(model, bundle, rng) -> list:
    """Patient-style queries: a few of a disease's own phenotypes (plus, in one variant, unrelated ones) and nothing else."""
    ids = list(model.ids)
    eligible = [d for d in ids if sum(w > 0 for w in bundle.records[d]["phenotypes"].values()) >= 10]
    chosen = sorted(rng.choice(eligible, min(SYMPTOM_DISEASES, len(eligible)), replace=False).tolist())
    all_terms = sorted({t for d in ids for t in bundle.records[d]["phenotypes"]})
    siblings = {}
    for a, b in model.relation_evidence.get("orphanet_siblings", {}):
        siblings.setdefault(a, set()).add(b); siblings.setdefault(b, set()).add(a)
    queries = []
    for d in chosen:
        phen = bundle.records[d]["phenotypes"]
        terms = sorted(t for t in phen if phen[t] > 0); weights = np.array([phen[t] for t in terms], dtype=float); weights /= weights.sum()
        for name, k, noise in SYMPTOM_VARIANTS:
            true = rng.choice(terms, k, replace=False, p=weights).tolist()
            extra = [t for t in rng.choice(all_terms, noise + 5, replace=False).tolist() if t not in phen][:noise]
            queries.append({"disease": d, "variant": name, "terms": true + extra, "n_true": k,
                            "siblings": sorted(siblings.get(d, ()))})
    return queries


def build_tasks(model, bundle) -> dict:
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
    sq = symptom_queries(model, bundle, np.random.default_rng(SEED + 1))
    return {
        "symptom_queries": sq,
        "n_modalities": dict(zip(ids, model.engine.available.sum(axis=1).astype(int).tolist())),
        "category": dict(zip(ids, model.top_categories)),
        "catalogue": ids,
        "names": dict(zip(ids, model.names)),
        "split": {i: assign_split(i) for i in ids},
        "paper_pairs": pairs,
        "relations": relations,
    }


def score_symptoms(tasks, model, bundle, ids) -> None:
    """v1: Jaccard of raw HPO sets (the only family a symptom-only query has); v2: the fusion refitted to the modalities present."""
    from data_sources import record_from_user_input
    import scipy.sparse as sp
    vocab: dict = {}
    rows, cols = [], []
    for i, d in enumerate(ids):
        for t in bundle.v1_sets[d]["hpo_ids"]:
            rows.append(i); cols.append(vocab.setdefault(t, len(vocab)))
    gallery = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(ids), len(vocab)))
    sizes = np.asarray(gallery.sum(axis=1)).ravel()
    v1, v2 = [], []
    for q in tasks["symptom_queries"]:
        known = [t for t in q["terms"] if t in vocab]
        vec = np.zeros(len(vocab), np.float32); vec[[vocab[t] for t in known]] = 1.0
        inter = gallery @ vec
        v1.append(np.divide(inter, np.maximum(len(known) + sizes - inter, 1.0)))
        record, _ = record_from_user_input({"name": "", "phenotypes": {t: 1.0 for t in q["terms"]}}, bundle.knowledge)
        v2.append(model.score_record(record)["logits"])
    folder = core.DATA / "scores" / "symptoms"; folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "v1.npy", np.array(v1, np.float32)); np.save(folder / "v2.npy", np.array(v2, np.float32))
    print(f"[v2] scored symptoms: {len(v1)} queries", flush=True)


def main() -> None:
    model = load_model(); bundle = data_sources.load_bundle()
    ids = list(model.ids)
    tasks = build_tasks(model, bundle)
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

    score_symptoms(tasks, model, bundle, ids)
    pair_queries = sorted({p[k] for p in tasks["paper_pairs"] for k in ("a", "b")})
    score_task("paper_pairs", pair_queries, (), ())
    kind_to_spec = {"orphanet_siblings": "orphanet_siblings", "shared_causal_gene": "shared_causal_gene", "shared_drug": "shared_drug"}
    for kind, spec_name in kind_to_spec.items():
        spec = BENCHMARK_SPECS[spec_name]
        score_task(f"rel_{kind}", list(tasks["relations"][kind]["queries"]), spec.masked, spec.v1_masked)


if __name__ == "__main__":
    main()
