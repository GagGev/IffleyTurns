"""Audit of the easy->hard pipeline (run.py): leakage, ground-truth validity (vs literature data), evaluation weaknesses.

  python easy_hard/audit.py            writes .data/easy_hard/audit_results.json and prints a report
"""
import csv
import json
import sys
import warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

sys.argv = sys.argv[:1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = run.ROOT
OUT = run.OUT
GROUPS = run.easy_groups()
ALL_EASY = None  # filled once from the first pair frame


def say(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ generic experiment runner
def experiment(d, easy_drop_groups=(), n_pairs=None, keep=False, seed=run.SEED):
    """Build ground truth + pairs for universe d (with d.split set), train gradient boosting on easy features and report
    held-out-disease metrics on natural (random) test pairs.  Uses the same recipe as run.py."""
    global ALL_EASY
    rows = list(d.itertuples(index=False))
    hf = run.HardFeatures(d)
    rng = np.random.default_rng(seed)
    tr = np.where(d.split.values == "train")[0]
    bi, bj = [], []
    while len(bi) < run.N_BACKGROUND:
        i, j = rng.choice(tr, 2, replace=False); bi.append(min(i, j)); bj.append(max(i, j))
    gt = run.GroundTruth(hf, np.array(bi), np.array(bj))
    n_pairs = n_pairs or run.PAIRS
    data = {s: run.sample_pairs(d, rows, s, n, rng, gt) for s, n in n_pairs.items()}
    easy_all = [c for c in data["train"].columns if c in GROUPS]
    ALL_EASY = easy_all
    easy = [c for c in easy_all if GROUPS[c] not in easy_drop_groups]
    w = data["train"].w.values
    m = run.fit("gradient_boosting", data["train"][easy].values, data["train"].target_similarity.values, w)
    out = {"n_diseases": int(len(d)), "easy_features_used": len(easy)}
    for s in ("validation", "test"):
        df = data[s]
        p = np.clip(m.predict(df[easy].values), 0, 1)
        r = (df.kind == "random").values
        out[s] = run.metrics(df.target_similarity.values[r], p[r])
    if keep:
        out["_objects"] = dict(d=d, rows=rows, hf=hf, gt=gt, data=data, model=m, easy=easy)
    return out


def fmt(m):
    return f"R2={m['r2']:6.3f}  Spearman={m['spearman']:.3f}"


# ------------------------------------------------------------------ family-grouped split
def sibling_purged_split(d, gmax=10, cmax=6, nmax=6):
    """Keep the hash split but REMOVE training diseases that have a close sibling in validation/test, where siblings share
    a rare gene (<=gmax carriers), a small Orphanet category (<=cmax) or a rare name token (<=nmax).  (A connected-component
    family split is impossible: gene sharing chains a third of all diseases into one component.)"""
    inv = defaultdict(list)
    for i, s in enumerate(d.h_gene):
        for g in s:
            inv[("g", g)].append(i)
    for i, s in enumerate(d.e_category):
        for c in s:
            inv[("c", c)].append(i)
    for i, s in enumerate(d.e_name):
        for t in s:
            inv[("n", t)].append(i)
    lim = {"g": gmax, "c": cmax, "n": nmax}
    nbr = defaultdict(set)
    for (k, _), v in inv.items():
        if 2 <= len(v) <= lim[k]:
            for a, b in combinations(v, 2):
                nbr[a].add(b); nbr[b].add(a)
    split = d.split.values.copy()
    held = {i for i in range(len(d)) if split[i] != "train"}
    purged = [i for i in range(len(d)) if split[i] == "train" and nbr[i] & held]
    split[purged] = "purged"
    return split, len(purged), nbr


# ------------------------------------------------------------------ literature (paper) data
def load_paper_pairs():
    """Pair-level literature evidence from both datasets.  Returns {frozenset(orpha ids): dict}."""
    def read(pairs_csv, dims_csv):
        P = list(csv.DictReader(open(pairs_csv)))
        D = list(csv.DictReader(open(dims_csv)))
        return P, D
    sources = [read(ROOT / "literature_review" / "literature_disease_pairs.csv", ROOT / "literature_review" / "paper_dimension_scores.csv"),
               read(ROOT / "literature_review" / "additional_runs" / "literature_disease_pairs_pairfirst.csv",
                    ROOT / "literature_review" / "additional_runs" / "paper_dimension_scores_pairfirst.csv")]
    dims = ["phenotype", "genetic", "mechanism", "therapeutic", "natural_history", "diagnostic_confusability", "comorbidity"]
    papers, seen = defaultdict(list), set()
    for P, D in sources:
        for k, (p, dd) in enumerate(zip(P, D)):
            a, b = p["disease_a_orpha_id"].strip(), p["disease_b_orpha_id"].strip()
            if not a or not b or ";" in a or ";" in b or a == b:
                continue
            key = (frozenset((a, b)), p["pmid"])
            if key in seen:
                continue
            seen.add(key)
            rec = {"sim": float(p["similarity_score"] or 0), "rel": p["relationship"]}
            for dim in dims:
                w, s = dd.get(f"w_{dim}", ""), dd.get(f"s_{dim}", "")
                rec[dim] = (float(w), float(s)) if w not in ("", None) and s not in ("", None) and float(w) > 0 else None
            papers[frozenset((a, b))].append(rec)
    out = {}
    for pair, rs in papers.items():
        e = {"n_papers": len(rs), "paper_sim": float(np.mean([r["sim"] for r in rs])), "unrelated": any(r["rel"] == "unrelated" for r in rs)}
        for dim in dims:
            v = [r[dim] for r in rs if r[dim]]
            e[f"paper_{dim}"] = sum(w * s for w, s in v) / sum(w for w, _ in v) if v else np.nan
        out[pair] = e
    return out


def bootstrap_ci(x, y, n=1000, seed=1):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(x)); vals = []
    for _ in range(n):
        b = rng.choice(idx, len(idx))
        if np.std(x[b]) > 0 and np.std(y[b]) > 0:
            vals.append(spearmanr(x[b], y[b])[0])
    return np.percentile(vals, [2.5, 97.5])


def main():
    res = {}
    say("=" * 100, "\nAUDIT OF easy_hard/run.py\n", "=" * 100)

    # ------------------------------------------------------------- 0. baseline and universe
    d0 = run.load_diseases()
    base = experiment(d0, keep=True)
    O = base.pop("_objects")
    res["E0_baseline"] = base
    say(f"\n[E0] v2 as shipped: {base['n_diseases']} diseases | validation {fmt(base['validation'])} | test {fmt(base['test'])}")

    allD = pd.read_parquet(ROOT / ".data/features/diseases.parquet")
    allD = allD[(allD.disorder_group == "Disorder") & allD.disorder_type.isin(["Disease", "Malformation syndrome"])]
    inc = allD.orpha_id.isin(set(d0.orpha_id))
    def top_systems(df):
        s = df.body_system_names.explode().dropna()
        s = s[~s.isin(run.GENERIC_BODY_SYSTEMS)]
        return (s.value_counts() / len(df)).head(6)
    uni = pd.concat([top_systems(allD[inc]).rename("included"), top_systems(allD[~inc]).rename("excluded")], axis=1).fillna(0)
    res["universe_bias"] = {"orphanet_disorders": int(len(allD)), "in_universe": int(inc.sum())}
    say(f"\n[universe] only {inc.sum()} of {len(allD)} Orphanet disorders ({inc.mean():.0%}) have HPO + genes + pathways + GO.")
    say("  top body-system share  included vs excluded:\n" + uni.round(2).to_string())

    # ------------------------------------------------------------- 1. leakage experiments
    say("\n" + "-" * 100 + "\n1. LEAKAGE\n" + "-" * 100)
    # 1a. inheritance / onset terms live inside the 'hard' HPO set
    cnt = d0.h_hpo_raw.map(len)
    clean_cnt = run.load_diseases(clean_hpo=True).h_hpo_raw.map(len)
    say(f"1a. 'Hard' HPO sets contain inheritance/onset/modifier terms (mean {cnt.mean():.1f} raw terms/disease before cleaning, "
        f"{clean_cnt.mean():.1f} after keeping only 'Phenotypic abnormality').")
    pc = run.load_diseases(clean_hpo=True)
    E1 = experiment(pc)
    say(f"    E1 ground truth with clean phenotype set: test {fmt(E1['test'])}   (vs E0 {fmt(base['test'])})")
    res["E1_clean_hpo"] = E1
    # 1b. gene sets from low-confidence Open Targets associations + names containing gene symbols
    pcg = run.load_diseases(clean_hpo=True, curated_genes=True)
    E2 = experiment(pcg)
    say(f"1b. curated-only genes (drops Open Targets 'overall' rows; mean genes/disease {d0.h_gene.map(len).mean():.1f} -> {pcg.h_gene.map(len).mean():.1f}), "
        f"{E2['n_diseases']} diseases:\n    E2 = E1 + curated genes: test {fmt(E2['test'])}")
    res["E2_clean_hpo_curated_genes"] = E2
    # 1c. easy features that share a source with the ground truth
    E3 = experiment(pcg, easy_drop_groups=("name", "inheritance"))
    say(f"1c. drop easy features that duplicate hard sources (name tokens: gene symbols appear in "
        f"{np.mean([any(g.lower() in n.lower().split() or g in n for g in gs) for n, gs in zip(d0.name, d0.h_gene)]):.1%} of names; "
        f"inheritance: ClinGen MOI is stored with the gene-disease record)\n    E3 = E2 - name - inheritance: test {fmt(E3['test'])}")
    res["E3_drop_name_inheritance"] = E3
    # 1d. sibling leakage
    split, n_purged, nbr = sibling_purged_split(pcg)
    held = np.where(pcg.split.values != "train")[0]
    te = np.where(pcg.split.values == "test")[0]
    with_sib = np.mean([any(pcg.split.values[j] == "train" for j in nbr[i]) for i in te])
    say(f"1d. sibling leakage. In the shipped hash split {with_sib:.0%} of test diseases have a close sibling (rare shared gene, small category "
        f"or rare name token) in TRAINING.\n    Purging those {n_purged} training diseases ({n_purged / (pcg.split.values == 'train').sum():.0%} of train):")
    dfam = pcg.copy(); dfam["split"] = split
    E4 = experiment(dfam, easy_drop_groups=("name", "inheritance"))
    say(f"    E4 = E3 + sibling-purged training set: test {fmt(E4['test'])}")
    res["E4_sibling_purged"] = E4
    res["test_diseases_with_train_sibling_shipped_split"] = float(with_sib)

    # ------------------------------------------------------------- 2. confounds
    say("\n" + "-" * 100 + "\n2. CONFOUNDS IN THE TARGET\n" + "-" * 100)
    data = O["data"]; d = O["d"]; easy = O["easy"]
    cnt_hpo = d.h_hpo_raw.map(len).values; cnt_gene = d.h_gene.map(len).values
    cols = {}
    for s, df in data.items():
        gi, gj = df.gi.values, df.gj.values
        cols[s] = pd.DataFrame({"hpo_sum": np.log1p(cnt_hpo[gi]) + np.log1p(cnt_hpo[gj]), "hpo_diff": np.abs(np.log1p(cnt_hpo[gi]) - np.log1p(cnt_hpo[gj])),
                                "gene_sum": np.log1p(cnt_gene[gi]) + np.log1p(cnt_gene[gj]), "gene_diff": np.abs(np.log1p(cnt_gene[gi]) - np.log1p(cnt_gene[gj]))})
    w = data["train"].w.values
    mc = HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05, random_state=run.SEED).fit(cols["train"].values, data["train"].target_similarity.values, sample_weight=w)
    te = data["test"]; r = (te.kind == "random").values
    cm = run.metrics(te.target_similarity.values[r], np.clip(mc.predict(cols["test"].values), 0, 1)[r])
    say(f"2a. annotation-count-only baseline (how well-annotated each disease is, no clinical info): test {fmt(cm)}  "
        f"-> compare E0 {fmt(base['test'])}")
    res["count_only_baseline"] = cm
    uni = {c: float(spearmanr(te[c].values[r], te.target_similarity.values[r])[0]) for c in easy if c.endswith("_sim") or c in ("same_parent", "same_type")}
    best = sorted(uni.items(), key=lambda kv: -abs(kv[1]))[:5]
    say("2b. single easy features vs target (Spearman, natural test pairs): " + ", ".join(f"{k} {v:.3f}" for k, v in best) +
        f"\n    full model {base['test']['spearman']:.3f}  -> the 12-feature model adds {base['test']['spearman'] - best[0][1]:+.3f} over the single best feature")
    res["single_feature_spearman"] = uni
    t = data["train"][data["train"].kind == "random"]
    say(f"2c. gene vs pathway overlap are not independent: Spearman(raw gene, raw pathway) on natural train pairs = "
        f"{spearmanr(t.raw_gene, t.raw_pathway)[0]:.2f}; share of natural pairs with ANY gene overlap = {(t.raw_gene > 0).mean():.2%} "
        f"(so the equal-weight 'genetic' domain is almost constant on natural pairs)")

    # ------------------------------------------------------------- 3. ground truth vs literature
    say("\n" + "-" * 100 + "\n3. GROUND TRUTH VS LITERATURE (paper-derived scores)\n" + "-" * 100)
    papers = load_paper_pairs()
    idx_of = {o: i for i, o in enumerate(d.orpha_id)}
    use = [(p, e) for p, e in papers.items() if all(o in idx_of for o in p) and len(p) == 2]
    say(f"{len(papers)} literature pairs with unambiguous ORPHA ids; {len(use)} have both diseases in the v2 universe.")
    gi = np.array([idx_of[sorted(p)[0]] for p, _ in use]); gj = np.array([idx_of[sorted(p)[1]] for p, _ in use])
    sc = run.pair_frame(O["rows"], gi, gj, O["gt"])
    lit = pd.DataFrame([e for _, e in use])
    sc["pred_model"] = np.clip(O["model"].predict(sc[O["easy"]].values), 0, 1)
    v1 = []
    for i, j in zip(gi, gj):
        v1.append(0.6 * run.jaccard(O["rows"][i].h_hpo_raw, O["rows"][j].h_hpo_raw) + 0.4 * run.jaccard(O["rows"][i].h_gene, O["rows"][j].h_gene))
    sc["v1_exact_jaccard"] = v1
    sc["easy_mean_sim"] = sc[[c for c in O["easy"] if c.endswith("_sim")]].mean(axis=1)
    tab = {}
    say("3a. Spearman between our scores and the paper-derived overall similarity (per pair; n = %d):" % len(lit))
    for name in ["target_similarity", "dom_phenotype", "dom_genetic", "dom_mechanism", "v1_exact_jaccard", "pred_model", "easy_mean_sim"]:
        rho = spearmanr(sc[name], lit.paper_sim)[0]; lo, hi = bootstrap_ci(sc[name].values, lit.paper_sim.values)
        tab[name] = {"rho": float(rho), "ci95": [float(lo), float(hi)]}
        say(f"    {name:20} rho={rho:6.3f}  95% CI [{lo:.2f}, {hi:.2f}]")
    res["paper_overall_corr"] = tab
    say("3b. per-dimension agreement (our domain score vs paper score for the SAME dimension):")
    dimtab = {}
    for dom, pdim in (("dom_phenotype", "paper_phenotype"), ("dom_genetic", "paper_genetic"), ("dom_mechanism", "paper_mechanism"), ("dom_therapeutic", "paper_therapeutic")):
        ok = sc[dom].notna().values & lit[pdim].notna().values
        if ok.sum() >= 8:
            rho = spearmanr(sc[dom].values[ok], lit[pdim].values[ok])[0]
            dimtab[dom] = {"rho": float(rho), "n": int(ok.sum())}
            say(f"    {dom:16} vs {pdim:18} n={ok.sum():4}  rho={rho:6.3f}")
        else:
            say(f"    {dom:16} vs {pdim:18} n={ok.sum():4}  (too few)")
    res["paper_per_dimension_corr"] = dimtab
    # known-related (literature) pairs vs random pairs
    rnd = data["test"][data["test"].kind == "random"]
    auc = {}
    for name in ["target_similarity", "dom_phenotype", "dom_genetic", "dom_mechanism", "pred_model"]:
        rv = rnd[name].values if name in rnd.columns else np.clip(O["model"].predict(rnd[O["easy"]].values), 0, 1)
        y = np.r_[np.ones(len(sc)), np.zeros(len(rv))]; v = np.r_[sc[name].values, rv]
        ok = ~np.isnan(v); auc[name] = float(roc_auc_score(y[ok], v[ok]))
    say("3c. AUC: literature pairs (co-studied, i.e. believed related) vs random pairs: " + ", ".join(f"{k} {v:.3f}" for k, v in auc.items()))
    res["paper_vs_random_auc"] = auc
    # fit domain weights to paper data with disease-grouped CV
    X = sc[["dom_phenotype", "dom_genetic", "dom_mechanism"]].fillna(0).values; yv = lit.paper_sim.values
    parent = {}
    def f(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for (p, _), a, b in zip(use, gi, gj):
        parent[f(a)] = f(b)
    comp = np.array([f(a) for a in gi]); ucomp = np.unique(comp)
    rng = np.random.default_rng(7); folds = dict(zip(ucomp, rng.integers(0, 5, len(ucomp))))
    fold = np.array([folds[c] for c in comp]); oof = np.zeros(len(yv))
    for k in range(5):
        trn, tst = fold != k, fold == k
        wts, _ = nnls(np.c_[X[trn], np.ones(trn.sum())], yv[trn]); oof[tst] = np.c_[X[tst], np.ones(tst.sum())] @ wts
    wall, _ = nnls(np.c_[X, np.ones(len(X))], yv)
    cv_rho = spearmanr(oof, yv)[0]; eq_rho = spearmanr(X.mean(axis=1), yv)[0]
    say(f"3d. weights fitted to paper data (NNLS, component-grouped 5-fold CV): out-of-fold rho {cv_rho:.3f} vs equal-weight rho {eq_rho:.3f}; "
        f"weights (phenotype, genetic, mechanism, intercept) = {np.round(wall, 3).tolist()}")
    res["paper_fitted_weights"] = {"weights": wall.tolist(), "cv_rho": float(cv_rho), "equal_rho": float(eq_rho), "n_pairs": len(yv)}

    # ------------------------------------------------------------- 4. evaluation variability
    say("\n" + "-" * 100 + "\n4. EVALUATION VARIABILITY (E0 recipe, different disease-hash salts)\n" + "-" * 100)
    runs = []
    for k in range(4):
        run.SALT = f"audit-salt-{k}"
        dk = run.load_diseases(); dk["split"] = dk.orpha_id.map(run.split_of)
        runs.append(experiment(dk, n_pairs={"train": 40000, "validation": 4000, "test": 8000})["test"])
        say(f"    salt {k}: {fmt(runs[-1])}")
    run.SALT = "easy-hard-v1"
    say(f"    mean R2 {np.mean([r['r2'] for r in runs]):.3f} +- {np.std([r['r2'] for r in runs]):.3f} ; mean Spearman {np.mean([r['spearman'] for r in runs]):.3f} +- {np.std([r['spearman'] for r in runs]):.3f}")
    res["salt_variability"] = runs
    json.dump(res, open(OUT / "audit_results.json", "w"), indent=2, default=float)
    say("\nsaved", OUT / "audit_results.json")


if __name__ == "__main__":
    main()
