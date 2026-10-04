"""Do the audit conclusions hold under top-10 'are the neighbours there?' metrics?  Also re-checks v2's metric code independently."""
import sys, warnings
from pathlib import Path
import numpy as np
sys.argv = sys.argv[:1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_v2 as A
from metrics import METRICS, query_metrics
from scoring import UniformFusion
warnings.filterwarnings("ignore")
M = {m: METRICS.index(m) for m in METRICS}

def manual(scores, positives):
    """Independent re-implementation (no shared code): rank by score, ties broken by index."""
    order = np.argsort(-scores, kind="stable")
    top = positives[order[:10]]
    ranks = np.flatnonzero(positives[order]) + 1
    n_pos = positives.sum()
    return dict(hits10=float(top.any()), p10=top.sum() / 10, ap=float(np.mean(np.arange(1, n_pos + 1) / ranks)), recall50=float((ranks <= 50).sum() / n_pos))

ctx, relations, train, prod = A.setup()
uni = UniformFusion(2.0)
top = ctx.bundle.meta["top_category"].to_dict()

print("1. METRIC CODE CHECK (v2 query_metrics vs independent re-implementation, 400 random queries per benchmark)")
for rel in relations.values():
    qs = A.test_queries(ctx, rel)[:400]
    gids, sc = A.score_matrix(ctx, rel, qs, prod)
    ours, theirs = [], []
    for i, q in enumerate(qs):
        valid = gids != q
        pos = np.isin(gids, list(rel.neighbors[q]))[valid]
        s = sc[i][valid]
        v = query_metrics(s, pos, np.random.default_rng(0))
        m = manual(s.astype(np.float64), pos)
        ours.append([v[M["hits@10"]], v[M["precision@10"]], v[M["map"]], v[M["recall@50"]]])
        theirs.append([m["hits10"], m["p10"], m["ap"], m["recall50"]])
    d = np.abs(np.array(ours) - np.array(theirs)).max(axis=0)
    print(f"   {rel.name:20} max |difference| per query  hits@10 {d[0]:.3f}  precision@10 {d[1]:.3f}  MAP {d[2]:.3f}  recall@50 {d[3]:.3f}   (mean P@10: v2 {np.mean([o[1] for o in ours]):.4f} vs manual {np.mean([t[1] for t in theirs]):.4f})")

print("\n2. TOP-10 METRICS FOR EVERY MODEL AND BASELINE  (test queries; Hits@10 = share of queries with >=1 related disease in the top 10; P@10 = share of the top 10 that are related)")
models = {"logistic_fusion": prod, "uniform_mean": uni}
variants = {"strict": (("ontology", "name", "text"), models)}
rows = []
for rel in relations.values():
    qs = A.test_queries(ctx, rel)
    r = RE = A.RE.evaluate(ctx, rel, qs, models, variants=variants)
    garr = np.array(sorted(rel.eligible)); pop = np.array([len(rel.neighbors.get(d, ())) for d in garr], float)
    cat = np.array([top[d] for d in garr])
    base = {"same top-level category": [], "candidate popularity (oracle)": []}
    for q in qs:
        valid = garr != q
        pos = np.isin(garr, list(rel.neighbors[q]))[valid]
        for k, v in (("same top-level category", (cat == top[q]).astype(float)), ("candidate popularity (oracle)", pop)):
            base[k].append(query_metrics(v[valid], pos, np.random.default_rng(0)))
    print(f"\n   {rel.name} ({len(qs)} queries, {np.mean([len(rel.neighbors[q]) for q in qs]):.1f} related diseases per query on average)")
    print(f"   {'model':34} Hits@10  P@10    nDCG@10  MAP")
    for name in ("random", "v1_weighted_jaccard", "uniform_mean", "logistic_fusion", "logistic_fusion_strict"):
        a = r["all"][name]
        print(f"   {name:34} {a[:, M['hits@10']].mean():.3f}    {a[:, M['precision@10']].mean():.3f}   {a[:, M['ndcg@10']].mean():.3f}    {a[:, M['map']].mean():.3f}")
    for k, v in base.items():
        a = np.array(v)
        print(f"   {k:34} {a[:, M['hits@10']].mean():.3f}    {a[:, M['precision@10']].mean():.3f}   {a[:, M['ndcg@10']].mean():.3f}    {a[:, M['map']].mean():.3f}")
