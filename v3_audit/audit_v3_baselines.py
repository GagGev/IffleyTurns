"""Non-neural baselines on v3's own prospective task (same queries, exclusions, galleries and metric as v3/run_evaluation.py).

Shows how far trivial rules get, i.e. the bar a neural model + stacker must clear.  Needs no PyTorch.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "v3")); warnings.filterwarnings("ignore")
from world import load_world, make_task
from features import StaticSimilarity, TemporalFeatures
from baselines import V1Baseline
from metrics import query_metrics, METRICS
MAP = METRICS.index("map"); AUC = METRICS.index("auc"); H10 = METRICS.index("hits@10"); P10 = METRICS.index("precision@10"); ND = METRICS.index("ndcg@10")

w = load_world(); sim = StaticSimilarity.from_world(w); v1 = V1Baseline(w.bundle.v1_sets, w.ids)
FOLDS = {"validation": ("2014-01-01", "2018-01-01"), "test": ("2018-01-01", None)}
out = {}
for fold, (a, b) in FOLDS.items():
    task = make_task(w, fold, pd.Timestamp(a), pd.Timestamp(b) if b else None)
    feats = TemporalFeatures(w, task.snapshot, sim, None)
    queries = task.queries(np.arange(w.n), w); warm = task.snapshot.warm
    names = ["random", "popularity (log #designations)", "popularity + graph degree", "adamic_adar", "drug mechanism", "v1 weighted Jaccard",
             "static uniform mean of cosines", "same top-level category", "static mean + popularity"]
    rec = {n: {"full": [], "warm": []} for n in names}; meta = []
    rng = np.random.default_rng(0)
    for s in range(0, len(queries), 64):
        rows = queries[s:s + 64]; B = feats.block(rows)
        sims = np.stack([B[f"sim_{m}"] for m in sim.modalities], -1)
        static_mean = np.nan_to_num(np.nanmean(sims, axis=-1), nan=0.0)
        pop = np.broadcast_to(feats.nodes.log_designations[None, :], (len(rows), w.n))
        sc = {"random": rng.random((len(rows), w.n)), "popularity (log #designations)": pop,
              "popularity + graph degree": pop + 1e-3 * np.log1p(feats.nodes.degree)[None, :], "adamic_adar": B["adamic_adar"],
              "drug mechanism": B["target_cosine"] + B["gene_target"] + 0.5 * B["target_pathway_cosine"],
              "v1 weighted Jaccard": v1.block(rows), "static uniform mean of cosines": static_mean,
              "same top-level category": B["same_category"], "static mean + popularity": static_mean * 5 + pop}
        excluded = task.excluded(rows, w).toarray(); positive = (task.positives[rows].toarray() > 0) & ~excluded
        for i, q in enumerate(rows):
            for gal, members in (("full", np.ones(w.n, bool)), ("warm", warm)):
                cand = members & ~excluded[i]; pos = positive[i] & cand
                if not pos.any() or not (cand & ~pos).any(): continue
                for n in names:
                    rec[n][gal].append(query_metrics(np.asarray(sc[n][i])[cand], pos[cand], np.random.default_rng(q)))
                if gal == "full" and n == names[-1]:
                    meta.append((bool(warm[q]), bool(w.is_group[q])))
    out[fold] = {n: {g: float(np.mean([x[MAP] for x in v])) for g, v in r.items()} for n, r in rec.items()}
    top10 = {n: {g: (float(np.mean([x[H10] for x in v])), float(np.mean([x[P10] for x in v])), float(np.mean([x[ND] for x in v]))) for g, v in r.items()} for n, r in rec.items()}
    print(f"\n{fold} fold ({a} -> {b or 'end'}): {len(queries)} queries, {int(warm.sum())} warm nodes, {len(task.relations)} new relations")
    print(f"  {'model':34} FULL gallery: MAP  Hits@10  P@10  nDCG@10 | WARM gallery: MAP  Hits@10  P@10  nDCG@10")
    for n in names:
        f, wm = top10[n]['full'], top10[n]['warm']
        print(f"  {n:34} {out[fold][n]['full']:.3f}  {f[0]:.3f}   {f[1]:.3f}  {f[2]:.3f}   |   {out[fold][n]['warm']:.3f}  {wm[0]:.3f}   {wm[1]:.3f}  {wm[2]:.3f}")
    m = np.array(meta)
    print(f"  queries: {int(m[:,0].sum())} warm / {int((~m[:,0]).sum())} cold; {int(m[:,1].sum())} are group nodes")
