"""Independent audit of v2/ (does not modify it).  Reuses v2's own code so numbers are comparable to its REPORT.md.

  python v2_audit/audit_v2.py        (needs `python v2/run_evaluation.py` to have been run once for selected_config.json)

Probes
  X1  agreement with the literature-derived pair scores (never used by v2)
  X2  strict-independence test: also hide ontology/name/text, and performance by Orphanet distance of the positives
  X3  trivial baselines (same top-level category, candidate richness, candidate popularity)
  X4  leave-one-benchmark-out transfer
  X5  Orphanet benchmark without parent->subtype pairs
  X6  label-noise check: are v2's "false positives" supported by another curated relation?
  X7  sensitivity to the disease split
  X8  per-query heterogeneity (annotation richness, number of positives, hub genes)
"""
import csv
import dataclasses
import json
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "v2"))
warnings.filterwarnings("ignore")

import run_evaluation as RE  # noqa: E402
from benchmarks import OrphanetIndex, build_relations  # noqa: E402
from common import assign_split  # noqa: E402
from data_sources import load_bundle  # noqa: E402
from metrics import METRICS, query_metrics  # noqa: E402
from modalities import DiseaseEncoder  # noqa: E402
from scoring import SimilarityEngine, TrainingSet, UniformFusion, V1Baseline, build_training_set, fit_logistic  # noqa: E402

OUT = ROOT / ".data" / "v2_audit"
OUT.mkdir(parents=True, exist_ok=True)
MAP = METRICS.index("map")
SELECTED = json.loads((ROOT / "v2" / ".data" / "evaluation" / "selected_config.json").read_text())
C = SELECTED["logistic_params"]["C"]
RES = {}


def say(*a):
    print(*a, flush=True)


def setup(salt=None):
    bundle = load_bundle()
    ids = sorted(bundle.records)
    split = bundle.meta["split"].to_dict() if salt is None else {d: assign_split(d, salt) for d in ids}
    train_ids = {d for d in ids if split[d] == "train"}
    enc = DiseaseEncoder(bundle.knowledge).fit([bundle.records[d] for d in sorted(train_ids)], [bundle.records[d] for d in ids])
    engine = SimilarityEngine(enc.transform([bundle.records[d] for d in ids]), ids)
    ctx = RE.Context(bundle, engine, V1Baseline(bundle.v1_sets, ids), OrphanetIndex(bundle), split)
    relations = build_relations(bundle)
    train = build_training_set(engine, list(relations.values()), train_ids, 10, 0)
    prod = fit_logistic(train, engine.modalities, C, True)
    prod.name = "logistic_fusion"
    return ctx, relations, train, prod


def test_queries(ctx, relation):
    return relation.queries(RE.split_ids(ctx, "test"))


def mean_map(result, name):
    return float(result["all"][name][:, MAP].mean())


# ------------------------------------------------------------------ literature data
def load_paper_pairs():
    srcs = [(ROOT / "literature_review" / "literature_disease_pairs.csv", ROOT / "literature_review" / "paper_dimension_scores.csv"),
            (ROOT / "literature_review" / "additional_runs" / "literature_disease_pairs_pairfirst.csv",
             ROOT / "literature_review" / "additional_runs" / "paper_dimension_scores_pairfirst.csv")]
    dims = ["phenotype", "genetic", "mechanism", "therapeutic", "natural_history", "diagnostic_confusability", "comorbidity"]
    papers, seen = defaultdict(list), set()
    for pf, df in srcs:
        for p, dd in zip(csv.DictReader(open(pf)), csv.DictReader(open(df))):
            a, b = p["disease_a_orpha_id"].strip(), p["disease_b_orpha_id"].strip()
            if not a or not b or ";" in a or ";" in b or a == b or (frozenset((a, b)), p["pmid"]) in seen:
                continue
            seen.add((frozenset((a, b)), p["pmid"]))
            rec = {"sim": float(p["similarity_score"] or 0)}
            for d in dims:
                w, s = dd.get(f"w_{d}", ""), dd.get(f"s_{d}", "")
                rec[d] = (float(w), float(s)) if w not in ("", None) and s not in ("", None) and float(w) > 0 else None
            papers[frozenset((a, b))].append(rec)
    out = {}
    for pair, rs in papers.items():
        e = {"paper_sim": float(np.mean([r["sim"] for r in rs])), "n_papers": len(rs)}
        for d in dims:
            v = [r[d] for r in rs if r[d]]
            e[f"paper_{d}"] = sum(w * s for w, s in v) / sum(w for w, _ in v) if v else np.nan
        out[pair] = e
    return out


def boot_ci(x, y, n=1000, seed=1):
    rng = np.random.default_rng(seed); idx = np.arange(len(x)); vals = []
    for _ in range(n):
        b = rng.choice(idx, len(idx))
        if np.std(x[b]) > 0 and np.std(y[b]) > 0:
            vals.append(spearmanr(x[b], y[b])[0])
    return np.percentile(vals, [2.5, 97.5])


def fus(model, S, A):
    """v2 fusions expect (queries, gallery, modalities); score a flat list of pairs."""
    return model.score(S[None], A[None])[0]


def auc(pos, neg):
    r = rankdata(np.r_[pos, neg])
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def x1_literature(ctx, relations, train, prod):
    say("\n" + "=" * 100 + "\nX1. AGREEMENT WITH LITERATURE-DERIVED PAIR SCORES (never used by v2)\n" + "=" * 100)
    bundle, engine = ctx.bundle, ctx.engine
    papers = load_paper_pairs()
    use = [(sorted(p), e) for p, e in papers.items() if all(o in engine.index for o in p) and len(p) == 2]
    a_ids = [p[0] for p, _ in use]; b_ids = [p[1] for p, _ in use]
    lit = pd.DataFrame([e for _, e in use])
    S, A = engine.pair_features(a_ids, b_ids)
    uni = fus(UniformFusion(2.0), S, A)
    v1 = np.array([ctx.v1.block([a], [b], ())[0, 0] for a, b in zip(a_ids, b_ids)])
    score = {"v2 logistic (production)": fus(prod, S, A), "v2 uniform mean": uni, "v1 weighted Jaccard": v1}
    say(f"{len(papers)} literature pairs with unambiguous ORPHA ids; {len(use)} are in the v2 cohort.")
    say("Spearman with paper-derived overall similarity (95% bootstrap CI):")
    res = {}
    for name, v in score.items():
        rho = spearmanr(v, lit.paper_sim)[0]; lo, hi = boot_ci(np.asarray(v), lit.paper_sim.values)
        res[name] = {"rho": float(rho), "ci": [float(lo), float(hi)]}
        say(f"   {name:28} rho={rho:6.3f}  [{lo:.2f}, {hi:.2f}]")
    mod = {m: j for j, m in enumerate(engine.modalities)}
    say("Single modalities against the SAME-dimension paper score:")
    dimres = {}
    for m, pd_ in (("phenotype", "paper_phenotype"), ("gene", "paper_genetic"), ("pathway", "paper_mechanism"), ("drug", "paper_therapeutic")):
        ok = A[:, mod[m]] & lit[pd_].notna().values
        if ok.sum() >= 15:
            rho = spearmanr(S[ok, mod[m]], lit[pd_].values[ok])[0]
            dimres[m] = {"rho": float(rho), "n": int(ok.sum())}
            say(f"   {m:10} vs {pd_:18} n={ok.sum():4} rho={rho:6.3f}")
        else:
            say(f"   {m:10} vs {pd_:18} n={int(ok.sum()):4} (too few)")
    for m in ("text", "name", "ontology"):
        ok = A[:, mod[m]]
        dimres[f"{m}_vs_overall"] = {"rho": float(spearmanr(S[ok, mod[m]], lit.paper_sim.values[ok])[0]), "n": int(ok.sum())}
        say(f"   {m:10} vs overall paper similarity   n={int(ok.sum()):4} rho={dimres[m + '_vs_overall']['rho']:6.3f}")
    # held-out pairs only
    held = np.array([ctx.split[a] != "train" and ctx.split[b] != "train" for a, b in zip(a_ids, b_ids)])
    if held.sum() >= 20:
        rho = spearmanr(score["v2 logistic (production)"][held], lit.paper_sim.values[held])[0]
        say(f"   pairs whose diseases are both unseen by the fusion/encoders (val+test): n={int(held.sum())}, rho={rho:.3f}")
        res["held_out_only"] = {"n": int(held.sum()), "rho": float(rho)}
    # co-studied vs random pairs, and vs random pairs from the same top-level category
    rng = np.random.default_rng(3)
    ids = engine.ids; top = bundle.meta["top_category"].to_dict()
    by_cat = defaultdict(list)
    for d in ids:
        by_cat[top[d]].append(d)
    rnd_a = rng.choice(ids, 4000); rnd_b = rng.choice(ids, 4000)
    keep = rnd_a != rnd_b
    Sr, Ar = engine.pair_features(list(rnd_a[keep]), list(rnd_b[keep]))
    same_b = [rng.choice(by_cat[top[a]]) for a in a_ids]
    Sc, Ac = engine.pair_features(a_ids, same_b)
    lit_score = fus(prod, S, A)
    say(f"AUC of literature pairs vs random pairs: {auc(lit_score, fus(prod, Sr, Ar)):.3f}; vs random pairs from the SAME top-level category: "
        f"{auc(lit_score, fus(prod, Sc, Ac)):.3f}  (the second removes the trivial 'same organ system' signal)")
    res["auc_vs_random"] = auc(lit_score, fus(prod, Sr, Ar))
    res["auc_vs_same_category_random"] = auc(lit_score, fus(prod, Sc, Ac))
    RES["X1_literature"] = {"overall": res, "per_dimension": dimres, "n_pairs": len(use)}


# ------------------------------------------------------------------ strict independence
def x2_strict(ctx, relations, train, prod):
    say("\n" + "=" * 100 + "\nX2. STRICT INDEPENDENCE: hide ontology + name + text as well (they encode the same curated families)\n" + "=" * 100)
    uni = UniformFusion(2.0)
    out = {}
    models = {"logistic_fusion": prod, "uniform_mean": uni}
    variants = {"strict": (("ontology", "name", "text"), models), "noont": (("ontology",), models)}
    full = {}
    for rel in relations.values():
        qs = test_queries(ctx, rel)
        r = RE.evaluate(ctx, rel, qs, models, variants=variants, with_hard=rel.name != "orphanet_siblings")
        full[rel.name] = (r, qs)
        row = {n: mean_map(r, n) for n in ("random", "v1_weighted_jaccard", "logistic_fusion", "logistic_fusion_noont", "logistic_fusion_strict", "uniform_mean_strict")}
        if "hard" in r:
            row.update({f"hard_{n}": float(r["hard"][n][:, MAP].mean()) for n in ("random", "v1_weighted_jaccard", "logistic_fusion", "logistic_fusion_strict")})
        out[rel.name] = row
        masked = ", ".join(rel.spec.masked)
        say(f"{rel.name}  (benchmark already masks: {masked}; {len(qs)} test queries)")
        say("   MAP  random {random:.3f} | v1 {v1_weighted_jaccard:.3f} | v2 logistic {logistic_fusion:.3f} | + hide ontology {logistic_fusion_noont:.3f} | "
            "+ hide ontology,name,text {logistic_fusion_strict:.3f} | uniform strict {uniform_mean_strict:.3f}".format(**row))
        if "hard_random" in row:
            say("   cross-group positives only:  random {hard_random:.3f} | v1 {hard_v1_weighted_jaccard:.3f} | v2 {hard_logistic_fusion:.3f} | v2 strict {hard_logistic_fusion_strict:.3f}".format(**row))
    RES["X2_strict"] = out
    return full


# ------------------------------------------------------------------ baselines
def x3_baselines(ctx, relations, full):
    say("\n" + "=" * 100 + "\nX3. TRIVIAL BASELINES (query-independent or one-line rules)\n" + "=" * 100)
    top = ctx.bundle.meta["top_category"].to_dict()
    engine = ctx.engine
    richness = dict(zip(engine.ids, engine.available.sum(axis=1)))
    out = {}
    for rel in relations.values():
        r, qs = full[rel.name]
        gallery = sorted(rel.eligible)
        garr = np.array(gallery)
        pop = {d: len(rel.neighbors.get(d, ())) for d in gallery}
        rich = np.array([richness[d] for d in gallery], dtype=float)
        popularity = np.array([pop[d] for d in gallery], dtype=float)
        cat = np.array([top[d] for d in gallery])
        aps = {"same top-level category": [], "candidate annotation richness": [], "candidate popularity (# of its own relations)": [], "same category, then richness": []}
        for q in qs:
            valid = garr != q
            pos = np.isin(garr, list(rel.neighbors[q]))[valid]
            same = (cat == top[q]).astype(float)
            rng = np.random.default_rng(0)
            cand = {"same top-level category": same, "candidate annotation richness": rich,
                    "candidate popularity (# of its own relations)": popularity, "same category, then richness": same * 100 + rich}
            for k, v in cand.items():
                aps[k].append(query_metrics(v[valid], pos, rng)[MAP])
        row = {k: float(np.mean(v)) for k, v in aps.items()}
        row["v2_logistic"] = mean_map(r, "logistic_fusion"); row["random"] = mean_map(r, "random")
        out[rel.name] = row
        say(f"{rel.name:20} random {row['random']:.3f} | " + " | ".join(f"{k} {row[k]:.3f}" for k in aps) + f" | v2 {row['v2_logistic']:.3f}")
    RES["X3_baselines"] = out


# ------------------------------------------------------------------ leave one benchmark out
def x4_loo(ctx, relations, train, prod, full):
    say("\n" + "=" * 100 + "\nX4. LEAVE-ONE-BENCHMARK-OUT: fusion trained WITHOUT the benchmark it is tested on\n" + "=" * 100)
    out = {}
    for b, rel in enumerate(relations.values()):
        keep = train.benchmark != b
        w = train.weight[keep]
        sub = TrainingSet(S=train.S[keep], A=train.A[keep], y=train.y[keep], weight=w * keep.sum() / w.sum(), benchmark=train.benchmark[keep])
        m = fit_logistic(sub, ctx.engine.modalities, C, True); m.name = "loo"
        r = RE.evaluate(ctx, rel, full[rel.name][1], {"loo": m}, with_baselines=False)
        out[rel.name] = {"multi_task": mean_map(full[rel.name][0], "logistic_fusion"), "leave_benchmark_out": mean_map(r, "loo")}
        say(f"{rel.name:20} MAP multi-task {out[rel.name]['multi_task']:.3f} -> trained without this benchmark {out[rel.name]['leave_benchmark_out']:.3f}"
            f"   (random {mean_map(full[rel.name][0], 'random'):.3f})")
    RES["X4_leave_one_benchmark_out"] = out


def x5_subtypes(ctx, relations, prod, full):
    say("\n" + "=" * 100 + "\nX5. ORPHANET BENCHMARK: how much is trivial parent->subtype?\n" + "=" * 100)
    rel = relations["orphanet_siblings"]
    only_sub = {k for k, v in rel.evidence.items() if all(part.startswith("subtype of") for part in v.split("; "))}
    sib_only = {k for k in rel.evidence if k not in only_sub}
    say(f"{len(rel.evidence):,} related pairs: {len(only_sub):,} are subtype-only ({len(only_sub) / len(rel.evidence):.1%}); the rest are siblings under a shared parent")
    nb = defaultdict(set)
    for a, b in sib_only:
        nb[a].add(b); nb[b].add(a)
    nb = {d: n & rel.eligible for d, n in nb.items() if d in rel.eligible}
    rel2 = dataclasses.replace(rel, neighbors=nb, evidence={k: rel.evidence[k] for k in sib_only})
    qs = [q for q in test_queries(ctx, rel) if nb.get(q)]
    r = RE.evaluate(ctx, rel2, qs, {"logistic_fusion": prod})
    say(f"siblings only ({len(qs)} queries): MAP random {mean_map(r, 'random'):.3f} | v1 {mean_map(r, 'v1_weighted_jaccard'):.3f} | v2 {mean_map(r, 'logistic_fusion'):.3f}"
        f"   (all pairs: v2 {mean_map(full['orphanet_siblings'][0], 'logistic_fusion'):.3f})")
    RES["X5_subtypes"] = {"subtype_only_share": len(only_sub) / len(rel.evidence), "siblings_only_v2_map": mean_map(r, "logistic_fusion"), "queries": len(qs)}


# ------------------------------------------------------------------ label noise
def score_matrix(ctx, relation, queries, model):
    gallery = ctx.engine.gallery(sorted(relation.eligible))
    out = np.zeros((len(queries), len(gallery.ids)), dtype=np.float32)
    for s in range(0, len(queries), 128):
        chunk = list(queries[s:s + 128])
        Qm, Qa = ctx.engine.rows(chunk)
        S, A = ctx.engine.block(Qm, Qa, gallery, masked=relation.spec.masked)
        out[s:s + len(chunk)] = model.score(S, A)
    return np.array(gallery.ids), out


def x6_label_noise(ctx, relations, prod, full):
    say("\n" + "=" * 100 + "\nX6. LABEL NOISE: are v2's top-10 'false positives' actually related by another curated source?\n" + "=" * 100)
    bundle = ctx.bundle
    top = bundle.meta["top_category"].to_dict()
    out = {}
    for rel in relations.values():
        qs = full[rel.name][1][:600]
        gids, sc = score_matrix(ctx, rel, qs, prod)
        n_fp = n_other = n_gene = n_cat = 0
        for i, q in enumerate(qs):
            order = np.argsort(-sc[i])
            taken = 0
            for j in order:
                c = gids[j]
                if c == q:
                    continue
                if taken >= 10:
                    break
                taken += 1
                if c in rel.neighbors.get(q, ()):
                    continue
                n_fp += 1
                if any(c in o.neighbors.get(q, ()) for o in relations.values() if o is not rel):
                    n_other += 1
                if set(bundle.records[q]["genes"]) & set(bundle.records[c]["genes"]):
                    n_gene += 1
                if top[c] == top[q]:
                    n_cat += 1
        out[rel.name] = {"false_positives": n_fp, "supported_by_other_benchmark": n_other / n_fp, "share_curated_gene": n_gene / n_fp, "same_top_category": n_cat / n_fp}
        say(f"{rel.name:20} top-10 'errors' on {len(qs)} queries: {n_fp}; {n_other / n_fp:.0%} are positives in ANOTHER benchmark, "
            f"{n_gene / n_fp:.0%} share a curated gene, {n_cat / n_fp:.0%} sit in the same top-level category")
    RES["X6_label_noise"] = out


def x7_splits(prod_ref):
    say("\n" + "=" * 100 + "\nX7. SENSITIVITY TO THE DISEASE SPLIT (same recipe, other hash salts)\n" + "=" * 100)
    out = {}
    for salt in ("audit-split-A", "audit-split-B"):
        ctx, relations, train, prod = setup(salt)
        row = {}
        for rel in relations.values():
            r = RE.evaluate(ctx, rel, test_queries(ctx, rel), {"logistic_fusion": prod}, with_baselines=False)
            row[rel.name] = mean_map(r, "logistic_fusion")
        out[salt] = row
        say(f"{salt}: " + " | ".join(f"{k} {v:.3f}" for k, v in row.items()))
    say("shipped split:   " + " | ".join(f"{k} {v['v2_logistic']:.3f}" for k, v in RES["X3_baselines"].items()))
    RES["X7_splits"] = out


def x8_heterogeneity(ctx, relations, full):
    say("\n" + "=" * 100 + "\nX8. WHAT DRIVES THE AVERAGE? per-query heterogeneity and hub genes\n" + "=" * 100)
    engine = ctx.engine
    rich = dict(zip(engine.ids, engine.available.sum(axis=1)))
    out = {}
    for rel in relations.values():
        r, qs = full[rel.name]
        ap = r["all"]["logistic_fusion"][:, MAP]
        npos = np.array([len(rel.neighbors[q]) for q in qs])
        q_rich = np.array([rich[q] for q in qs])
        out[rel.name] = {"spearman_AP_vs_n_positives": float(spearmanr(ap, npos)[0]), "spearman_AP_vs_query_annotation_richness": float(spearmanr(ap, q_rich)[0]),
                         "share_queries_AP_below_0.05": float((ap < 0.05).mean()), "share_queries_AP_above_0.5": float((ap > 0.5).mean()),
                         "median_AP": float(np.median(ap)), "mean_AP": float(ap.mean())}
        say(f"{rel.name:20} mean AP {ap.mean():.3f} but median {np.median(ap):.3f}; {np.mean(ap < 0.05):.0%} of queries have AP<0.05 and {np.mean(ap > 0.5):.0%} have AP>0.5; "
            f"corr(AP, #positives)={out[rel.name]['spearman_AP_vs_n_positives']:+.2f}, corr(AP, query annotation richness)={out[rel.name]['spearman_AP_vs_query_annotation_richness']:+.2f}")
    rel = relations["shared_causal_gene"]
    genes = Counter(e.split("causal gene ")[1] for e in rel.evidence.values() for e in e.split("; "))
    pairs = sum(genes.values())
    big = sum(v for v in genes.values() if v >= 45)  # >=10 diseases share the gene
    say(f"hub genes: {len(genes)} genes generate {pairs:,} shared-gene pairs; genes carried by >=10 diseases account for {big / pairs:.0%} of pairs; "
        f"top genes: {genes.most_common(5)}")
    out["hub_genes"] = {"pairs": pairs, "share_from_genes_with_10_plus_diseases": big / pairs, "top": genes.most_common(5)}
    RES["X8_heterogeneity"] = out


def x9_graph():
    say("\n" + "=" * 100 + "\nX9. PRODUCTION GRAPH CLAIMS\n" + "=" * 100)
    path = ROOT / "v2" / ".data" / "graph" / "edges.csv"
    if not path.is_file():
        say("graph not built; skipped")
        return
    e = pd.read_csv(path, keep_default_na=False)
    sup = e.support.value_counts().to_dict()
    kinds = {c: int((e[c] != "").sum()) for c in ("known_orphanet_relation", "known_shared_causal_gene", "known_shared_trial_drug")}
    via_ont = e[(e.known_orphanet_relation != "")]
    say(f"{len(e):,} edges; support {sup}; curated by relation type {kinds}")
    say(f"{len(via_ont) / len(e):.0%} of ALL edges are Orphanet siblings/subtypes - but the production graph model USES the ontology and name modalities "
        f"(which define that label) as inputs, so 'curated' support here is largely circular.")
    for m in ("ontology", "name", "text", "phenotype", "pathway", "gene"):
        c = f"contrib_{m}"
        if c in e:
            say(f"   mean contribution of {m:9} to edge score: {pd.to_numeric(e[c]).mean():+.2f}")
    RES["X9_graph"] = {"support": sup, "by_relation": kinds, "share_orphanet_relation": len(via_ont) / len(e)}


def main():
    ctx, relations, train, prod = setup()
    say("production-style fusion, trained on train diseases only: coefficients")
    for c in sorted(prod.coefficients(), key=lambda r: -r["similarity_weight"]):
        say(f"   {c['modality']:12} similarity {c['similarity_weight']:+.2f}  availability {c['availability_weight']:+.2f}")
    RES["coefficients"] = prod.coefficients()
    x1_literature(ctx, relations, train, prod)
    full = x2_strict(ctx, relations, train, prod)
    x3_baselines(ctx, relations, full)
    x4_loo(ctx, relations, train, prod, full)
    x5_subtypes(ctx, relations, prod, full)
    x6_label_noise(ctx, relations, prod, full)
    x8_heterogeneity(ctx, relations, full)
    x9_graph()
    json.dump(RES, open(OUT / "audit_v2_results.json", "w"), indent=2, default=float)
    x7_splits(prod)
    json.dump(RES, open(OUT / "audit_v2_results.json", "w"), indent=2, default=float)
    say("\nsaved", OUT / "audit_v2_results.json")


if __name__ == "__main__":
    main()
