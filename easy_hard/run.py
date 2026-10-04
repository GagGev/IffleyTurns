"""Easy-feature -> hard-feature-ground-truth pipeline (no papers).

A researcher supplies EASY features for two diseases (things they already know).  A model turns them into a predicted
similarity.  That prediction is scored against a ground truth computed from HARD features (things that need curation,
sequencing or phenotyping to obtain).

GROUND TRUTH (hard features), built to be defensible rather than hand-tuned:
  * every hard feature is an information-content (IC) weighted Jaccard similarity between the two diseases' annotation
    sets, IC(t) = -ln(P(a disease is annotated with t)).  Sharing a rare phenotype / gene / pathway counts for more than
    sharing a ubiquitous one (the standard approach for ontology-based similarity, e.g. Resnik/Phenomizer-style IC);
  * hard features are grouped into four biological domains that mirror the project's similarity dimensions:
        phenotype    : HPO terms, propagated to ancestors
        genetic      : associated genes
        mechanism    : Reactome pathways of those genes (propagated to ancestors)  +  GO biological-process terms
        therapeutic  : ChEMBL molecules (only when BOTH diseases have any; otherwise the domain is simply not used)
  * each feature score is converted to its PERCENTILE among random background pairs (fraction of random pairs with a
    strictly smaller score).  This puts domains with very different base rates on one scale ("how unusual is this
    overlap?") so no domain dominates just because its raw numbers are bigger;
  * the ground truth is the EQUAL-weight mean of the available domains (no tuned weights).  The choice is checked by
    sensitivity analyses (alternative weightings, raw scores, PC1 weights) and by an external check: does the score
    recover pairs of diseases that Orphanet experts placed in the same small disease category?

EASY features (unchanged, 12 similarity features): body system, inheritance, onset (set + ordinal), prevalence,
disease categories, Mondo parents, ICD-10 block/chapter, disease-name overlap, same parent, same disease type.

Evaluation is by DISEASE, not by pair: diseases are hashed into train/validation/test and a pair is only used when both
diseases fall in the same split, so test diseases are never seen in training.

  python easy_hard/run.py
"""
import glob
import hashlib
import warnings
import json
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".data" / "easy_hard"

# ---------------------------------------------------------------- configuration
DOMAINS = {  # domain -> hard features; domain score = mean of its features' percentiles
    "phenotype": ["hpo"],
    "genetic": ["gene"],
    "mechanism": ["pathway", "go_bp"],
    "therapeutic": ["drug"],  # optional domain
}
CORE = ["phenotype", "genetic", "mechanism"]  # every disease in the universe has all three
EQUAL = {"phenotype": 1.0, "genetic": 1.0, "mechanism": 1.0}
ONSET_ORDER = ["Antenatal", "Neonatal", "Infancy", "Childhood", "Adolescent", "Adult", "Elderly"]
GENERIC_BODY_SYSTEMS = {  # tags carried by most Orphanet diseases; they carry no information
    "Rare genetic disease", "genetic, familial or congenital disease", "phenotype",
    "Rare developmental defect during embryogenesis",
    "Rare disorder potentially indicated for transplant or complication after transplantation",
}
NAME_STOPWORDS = {"syndrome", "disease", "of", "and", "with", "the", "due", "to", "deficiency", "disorder", "type"}
SPLIT_FRACTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}
PAIRS = {"train": 40000, "validation": 8000, "test": 8000}  # half random pairs, half "informative"
N_BACKGROUND = 100000
SALT = "easy-hard-v1"
SEED = 20261004
# ---------------------------------------------------------------------------------


def split_of(orpha_id: str) -> str:
    h = int(hashlib.sha256(f"{SALT}|{orpha_id}".encode()).hexdigest(), 16) / 16**64
    return "train" if h < SPLIT_FRACTIONS["train"] else (
        "validation" if h < SPLIT_FRACTIONS["train"] + SPLIT_FRACTIONS["validation"] else "test")


def to_set(x, drop=()):
    return frozenset(str(v) for v in (x if x is not None else []) if v not in ("", None) and v not in drop)


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


# ---------------------------------------------------------------- hard-feature annotation sets
def hpo_ancestors():
    parents, cur = {}, None
    for line in open(ROOT / ".data" / "databases" / "hpo" / "hp.obo"):
        if line.startswith("id: HP:"):
            cur = line[4:].strip(); parents[cur] = set()
        elif line.startswith("is_a: HP:") and cur:
            parents[cur].add(line[6:16])
        elif line.startswith("[Term]"):
            cur = None
    memo = {}

    def up(t):
        if t not in memo:
            memo[t] = set()
            for p in parents.get(t, ()):
                memo[t] |= {p} | up(p)
        return memo[t]
    return {t: up(t) - {"HP:0000001", "HP:0000118"} for t in parents}


def gene_annotations():
    """gene symbol -> (Reactome pathway ids incl. ancestors, GO biological-process ids), from Open Targets."""
    t = pd.concat(pd.read_parquet(f, columns=["approvedSymbol", "pathways", "go"])
                  for f in sorted(glob.glob(str(ROOT / ".data/databases/opentargets/target/*.parquet"))))
    react = pd.read_parquet(glob.glob(str(ROOT / ".data/databases/opentargets/reactome/*.parquet"))[0], columns=["id", "ancestors"])
    anc = {i: set(a) if a is not None else set() for i, a in zip(react.id, react.ancestors)}
    pw, go = {}, {}
    for sym, p, g in zip(t.approvedSymbol, t.pathways, t.go):
        if p is not None and len(p):
            s = pw.setdefault(sym, set())
            for q in p:
                s.add(q["pathwayId"]); s |= anc.get(q["pathwayId"], set())
        if g is not None and len(g):
            go.setdefault(sym, set()).update(x["id"] for x in g if x["aspect"] == "P")
    return pw, go


def union_over(genes, mapping):
    out = set()
    for g in genes:
        out |= mapping.get(g, set())
    return frozenset(out)


def name_tokens(name):
    return frozenset(w for w in re.findall(r"[a-z0-9]+", name.lower()) if w not in NAME_STOPWORDS and len(w) > 1)


def hpo_abnormality_terms():
    """HPO terms under 'Phenotypic abnormality' (HP:0000118): excludes mode-of-inheritance, onset and clinical-course terms."""
    parents, cur = {}, None
    for line in open(ROOT / ".data" / "databases" / "hpo" / "hp.obo"):
        if line.startswith("id: HP:"):
            cur = line[4:].strip(); parents[cur] = set()
        elif line.startswith("is_a: HP:") and cur:
            parents[cur].add(line[6:16])
        elif line.startswith("[Term]"):
            cur = None
    memo = {}

    def under(t):
        if t not in memo:
            memo[t] = t == "HP:0000118" or any(under(p) for p in parents.get(t, ()))
        return memo[t]
    return {t for t in parents if under(t)}


def curated_gene_sets():
    """orpha_id -> genes from curated sources only (HPO, Orphanet, ClinGen, ClinVar); drops Open Targets 'overall' scores."""
    g = pd.read_parquet(ROOT / ".data" / "features" / "genes.parquet", columns=["orpha_id", "gene_symbol", "source"])
    g = g[~g.source.str.startswith("opentargets")]
    return g.groupby("orpha_id").gene_symbol.agg(lambda x: frozenset(v for v in x if v)).to_dict()


def load_diseases(clean_hpo=False, curated_genes=False) -> pd.DataFrame:
    """clean_hpo: keep only 'Phenotypic abnormality' HPO terms in the ground truth (removes inheritance/onset terms that
    duplicate easy features).  curated_genes: use curated gene-disease sources only (drops low-confidence Open Targets rows)."""
    d = pd.read_parquet(ROOT / ".data" / "features" / "diseases.parquet")
    d = d[(d.disorder_group == "Disorder") & d.disorder_type.isin(["Disease", "Malformation syndrome"])].copy()
    anc = hpo_ancestors()
    pw, go = gene_annotations()
    genes = d.gene_symbols.map(to_set)
    if curated_genes:
        cg = curated_gene_sets()
        genes = d.orpha_id.map(lambda o: cg.get(o, frozenset()))
    hpo = d.hpo_ids.map(to_set)
    if clean_hpo:
        ok = hpo_abnormality_terms()
        hpo = hpo.map(lambda s: frozenset(t for t in s if t in ok))
    d["h_gene"] = genes
    d["h_hpo"] = hpo.map(lambda s: frozenset(s.union(*[anc.get(t, set()) | {t} for t in s])) if s else frozenset())
    d["h_pathway"] = genes.map(lambda s: union_over(s, pw))
    d["h_go_bp"] = genes.map(lambda s: union_over(s, go))
    d["h_drug"] = d.drug_ids.map(lambda x: frozenset(v for v in to_set(x) if v.startswith("CHEMBL")))
    d["h_hpo_raw"] = hpo
    for c in ("h_hpo", "h_gene", "h_pathway", "h_go_bp"):  # core domains must exist for every disease in the universe
        d = d[d[c].map(bool)]
    d = d.reset_index(drop=True)
    # ---- easy feature sets
    d["e_body_system"] = d.body_system_names.map(lambda x: to_set(x, GENERIC_BODY_SYSTEMS))
    d["e_inheritance"] = d.inheritance.map(to_set)
    d["e_onset"] = d.onset.map(to_set)
    d["log_prev"] = np.log10(d.prevalence_estimated_per_person.where(d.prevalence_estimated_per_person > 0))
    d["e_category"] = d.category_ids.map(to_set)
    d["e_parent"] = d.preferential_parent_ids.map(to_set)
    d["e_mondo_parent"] = d.ontology_parent_ids.map(to_set)
    d["e_icd_block"] = d.icd10_ids.map(lambda x: frozenset(str(v)[:3] for v in (x if x is not None else []) if v))
    d["e_icd_chapter"] = d.icd10_ids.map(lambda x: frozenset(str(v)[:1] for v in (x if x is not None else []) if v))
    d["e_name"] = d.name.map(name_tokens)
    d["onset_pos"] = d.onset.map(lambda x: np.mean([ONSET_ORDER.index(v) for v in (x if x is not None else []) if v in ONSET_ORDER])
                                 if any(v in ONSET_ORDER for v in (x if x is not None else [])) else np.nan)
    d["is_malformation"] = (d.disorder_type == "Malformation syndrome").astype(float)
    d["split"] = d.orpha_id.map(split_of)
    return d


# ---------------------------------------------------------------- IC-weighted hard similarity (vectorised)
class HardFeatures:
    def __init__(self, d):
        self.mats, self.ic, self.size, self.has = {}, {}, {}, {}
        feats = [f for fs in DOMAINS.values() for f in fs]
        for f in feats:
            sets = list(d[f"h_{f}"])
            vocab = {t: k for k, t in enumerate(sorted(set().union(*sets)))}
            rows, cols = zip(*[(i, vocab[t]) for i, s in enumerate(sets) for t in s]) if vocab else ([], [])
            m = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(sets), len(vocab)))
            has = np.asarray(m.sum(axis=1)).ravel() > 0
            df = np.asarray(m.sum(axis=0)).ravel()
            ic = -np.log((df + 1.0) / (has.sum() + 1.0))  # IC over diseases that have this feature at all
            self.mats[f], self.ic[f], self.has[f] = m, ic, has
            self.size[f] = np.asarray(m @ ic).ravel()

    def raw(self, f, i, j):
        """IC-weighted Jaccard of diseases i[k], j[k]: sum IC(shared) / sum IC(union)."""
        m = self.mats[f]
        inter = np.asarray(m[i].multiply(m[j]) @ self.ic[f]).ravel()
        union = self.size[f][i] + self.size[f][j] - inter
        return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


class GroundTruth:
    """raw IC-Jaccard -> percentile vs random background pairs -> domain means -> equal-weight composite."""

    def __init__(self, hf: HardFeatures, bg_i, bg_j):
        self.hf = hf
        self.bg = {f: np.sort(hf.raw(f, bg_i, bg_j)) for f in hf.mats}

    def score(self, i, j) -> pd.DataFrame:
        out = {}
        for dom, fs in DOMAINS.items():
            pcts = []
            for f in fs:
                r = self.hf.raw(f, i, j)
                p = np.searchsorted(self.bg[f], r, side="left") / len(self.bg[f])  # P(background < score); 0 for zero overlap
                avail = self.hf.has[f][i] & self.hf.has[f][j]
                out[f"raw_{f}"], out[f"pct_{f}"] = r, np.where(avail, p, np.nan)
                pcts.append(out[f"pct_{f}"])
            out[f"dom_{dom}"] = np.nanmean(np.vstack(pcts), axis=0) if len(pcts) > 1 else pcts[0]
        return pd.DataFrame(out)


def composite(df, core_weights, col_prefix="dom_"):
    """Weighted mean over AVAILABLE domains; the optional therapeutic domain gets the mean core weight."""
    w = dict(core_weights); w["therapeutic"] = float(np.mean(list(core_weights.values())))
    num = np.zeros(len(df)); den = np.zeros(len(df))
    for dom, wt in w.items():
        v = df[f"{col_prefix}{dom}"].values
        ok = ~np.isnan(v)
        num += np.where(ok, wt * np.nan_to_num(v), 0.0); den += np.where(ok, wt, 0.0)
    return num / den


# ---------------------------------------------------------------- easy features
def easy_groups():
    g = {"body_system_sim": "body_system", "body_system_avail": "body_system",
         "inheritance_sim": "inheritance", "inheritance_avail": "inheritance",
         "onset_sim": "onset", "onset_avail": "onset", "onset_ordinal_sim": "onset", "onset_ordinal_avail": "onset",
         "prevalence_sim": "prevalence", "prevalence_avail": "prevalence",
         "same_parent": "classification", "same_parent_avail": "classification", "same_type": "disease_type"}
    for n, grp in (("category", "classification"), ("mondo_parent", "classification"), ("icd_block", "icd10"),
                   ("icd_chapter", "icd10"), ("name", "name")):
        g[f"{n}_sim"] = grp; g[f"{n}_avail"] = grp
    return g


def easy_row(a, b):
    x = {}
    for name in ("body_system", "inheritance", "onset", "category", "mondo_parent", "icd_block", "icd_chapter", "name"):
        sa, sb = getattr(a, f"e_{name}"), getattr(b, f"e_{name}")
        x[f"{name}_sim"] = jaccard(sa, sb)
        x[f"{name}_avail"] = float(bool(sa) and bool(sb))
    ok = not (np.isnan(a.log_prev) or np.isnan(b.log_prev))
    x["prevalence_sim"] = 1.0 / (1.0 + abs(a.log_prev - b.log_prev)) if ok else 0.0
    x["prevalence_avail"] = float(ok)
    x["same_parent"] = float(bool(a.e_parent & b.e_parent))
    x["same_parent_avail"] = float(bool(a.e_parent) and bool(b.e_parent))
    x["same_type"] = float(a.is_malformation == b.is_malformation)
    ok = not (np.isnan(a.onset_pos) or np.isnan(b.onset_pos))
    x["onset_ordinal_sim"] = 1.0 / (1.0 + abs(a.onset_pos - b.onset_pos)) if ok else 0.0
    x["onset_ordinal_avail"] = float(ok)
    return x


def pair_frame(rows, idx_i, idx_j, gt: GroundTruth):
    easy = pd.DataFrame([easy_row(rows[i], rows[j]) for i, j in zip(idx_i, idx_j)])
    hard = gt.score(np.asarray(idx_i), np.asarray(idx_j))
    df = pd.concat([easy, hard], axis=1)
    df["target_similarity"] = composite(df, EQUAL)
    df["gi"], df["gj"] = idx_i, idx_j
    return df


def sample_pairs(d, rows, split, n, rng, gt):
    ids = np.where(d.split.values == split)[0]  # global indices of this split's diseases
    inv_g, inv_p = {}, {}
    for k in ids:
        for g in rows[k].h_gene:
            inv_g.setdefault(g, []).append(k)
        for t in rows[k].h_hpo_raw:
            inv_p.setdefault(t, []).append(k)
    cand = set()
    for v in inv_g.values():
        if len(v) <= 40:
            cand.update(combinations(v, 2))
    cnt = {}
    for v in inv_p.values():
        if len(v) <= 150:
            for i, j in combinations(v, 2):
                cnt[(i, j)] = cnt.get((i, j), 0) + 1
    cand.update(k for k, c in cnt.items() if c >= 3)
    cand = sorted(cand)
    pi = len(cand) / (len(ids) * (len(ids) - 1) / 2)  # share of ALL pairs that are "informative"
    n_inf = min(n // 2, len(cand))
    inf = [cand[i] for i in rng.choice(len(cand), n_inf, replace=False)]
    rnd = set()
    while len(rnd) < n - n_inf:
        i, j = rng.choice(ids, 2, replace=False)
        rnd.add((int(min(i, j)), int(max(i, j))))
    rnd = sorted(rnd)
    # informative pairs are oversampled (50% of the sample vs ~pi of all pairs): down-weight them in training so the
    # model is calibrated to the natural pair population; random pairs get weight 1
    w_inf = pi * (n - n_inf) / max(n_inf, 1)
    parts = []
    for kind, pairs, wt in (("informative", inf, w_inf), ("random", rnd, 1.0)):
        df = pair_frame(rows, [p[0] for p in pairs], [p[1] for p in pairs], gt)
        df["kind"], df["split"], df["w"] = kind, split, wt
        df["a"], df["b"] = [rows[i].orpha_id for i in df.gi], [rows[j].orpha_id for j in df.gj]
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------- evaluation helpers
def metrics(y, p):
    return dict(n=int(len(y)), r2=float(r2_score(y, p)), rmse=float(mean_squared_error(y, p) ** 0.5),
                spearman=float(spearmanr(y, p)[0]) if np.std(p) > 0 and np.std(y) > 0 else float("nan"))


def retrieval(d, rows, gt, model, easy_cols, k=10):
    """Each held-out test disease ranks every other test disease; compare with the true top-k by ground truth."""
    ids = np.where(d.split.values == "test")[0]
    pairs = np.array(list(combinations(range(len(ids)), 2)))
    df = pair_frame(rows, ids[pairs[:, 0]], ids[pairs[:, 1]], gt)
    sims = [c for c in easy_cols if c.endswith("_sim")]
    preds = {"model": model.predict(df[easy_cols].values), "easy_only_mean": df[sims].mean(axis=1).values,
             "random": np.random.default_rng(SEED).random(len(df))}
    truth_all = df.target_similarity.values
    ii, jj = pairs[:, 0], pairs[:, 1]
    res = {n: [] for n in preds}
    for q in range(len(ids)):
        m = (ii == q) | (jj == q)
        top_true = set(np.argsort(-truth_all[m])[:k])
        for n, p in preds.items():
            res[n].append(len(top_true & set(np.argsort(-p[m])[:k])) / k)
    return {"queries": len(ids), "k": k, **{f"precision_at_k_{n}": float(np.mean(v)) for n, v in res.items()}}


def new_model(kind):
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    return HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05, random_state=SEED)


def fit(kind, X, y, w):
    m = new_model(kind)
    m.fit(X, y, **({"ridge__sample_weight": w} if kind == "ridge" else {"sample_weight": w}))
    return m


def external_check(d, rows, gt, rng):
    """Does the ground truth recover pairs that Orphanet experts placed in the same small disease category (<=30 diseases)?
    Negatives are random pairs sharing no category.  Disease categories are an *easy* feature, not an input to the
    ground truth, so this validates the target independently of it."""
    groups = {}
    for i, r in enumerate(rows):
        for c in r.e_category:
            groups.setdefault(c, []).append(i)
    allpos = sorted({(a, b) for v in groups.values() if 2 <= len(v) <= 30 for a, b in combinations(sorted(v), 2)})
    pos = [allpos[k] for k in rng.choice(len(allpos), min(6000, len(allpos)), replace=False)]
    neg = set()
    while len(neg) < len(pos):
        i, j = rng.integers(0, len(rows), 2)
        if i != j and not (rows[i].e_category & rows[j].e_category):
            neg.add((int(min(i, j)), int(max(i, j))))
    neg = sorted(neg)
    idx_i = np.array([p[0] for p in pos + neg]); idx_j = np.array([p[1] for p in pos + neg])
    lab = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    s = gt.score(idx_i, idx_j)
    out = {}
    for name, v in {**{f"raw {f} (IC-Jaccard)": s[f"raw_{f}"].values for f in gt.hf.mats},
                    **{f"domain: {k}": s[f"dom_{k}"].values for k in DOMAINS},
                    "COMPOSITE (equal weights)": composite(s, EQUAL),
                    "composite, phenotype-heavy 2:1:1": composite(s, {"phenotype": 2, "genetic": 1, "mechanism": 1}),
                    "composite, genetic-heavy 1:2:1": composite(s, {"phenotype": 1, "genetic": 2, "mechanism": 1}),
                    "composite, mechanism-heavy 1:1:2": composite(s, {"phenotype": 1, "genetic": 1, "mechanism": 2})}.items():
        ok = ~np.isnan(v)
        out[name] = {"auc": float(roc_auc_score(lab[ok], v[ok])), "pairs_scored": int(ok.sum())}
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    d = load_diseases()
    rows = list(d.itertuples(index=False))
    hf = HardFeatures(d)
    print(f"{len(d)} diseases with phenotype, gene, pathway and GO annotations; per split:", d.split.value_counts().to_dict())
    print("hard-feature coverage (diseases with any annotation):", {f: int(h.sum()) for f, h in hf.has.items()})
    tr = np.where(d.split.values == "train")[0]
    bi, bj = [], []
    while len(bi) < N_BACKGROUND:
        i, j = rng.choice(tr, 2, replace=False); bi.append(min(i, j)); bj.append(max(i, j))
    gt = GroundTruth(hf, np.array(bi), np.array(bj))

    ext = external_check(d, rows, gt, np.random.default_rng(SEED + 1))
    print("\nEXTERNAL CHECK - AUC for recovering pairs in the same small Orphanet disease category (0.5 = chance):")
    for k, v in ext.items():
        print(f"  {k:36} AUC {v['auc']:.3f}  (n={v['pairs_scored']})")

    data = {s: sample_pairs(d, rows, s, n, rng, gt) for s, n in PAIRS.items()}
    for s, df in data.items():
        df.drop(columns=["gi", "gj"]).to_csv(OUT / f"pairs_{s}.csv", index=False)
    groups = easy_groups()
    easy_cols = [c for c in data["train"].columns if c in groups]
    y = {s: df.target_similarity.values for s, df in data.items()}
    w = data["train"].w.values
    rnd = {s: (df.kind == "random").values for s, df in data.items()}
    t = data["train"][data["train"].kind == "random"]
    print(f"\n{len(easy_cols)} easy columns ({len([c for c in easy_cols if c.endswith('_sim') or c in ('same_parent', 'same_type')])} similarity features)")
    print("ground truth on natural train pairs: mean {:.3f}, sd {:.3f}; therapeutic domain available for {:.2%} of pairs".format(
        t.target_similarity.mean(), t.target_similarity.std(), t.dom_therapeutic.notna().mean()))
    print("correlation between domains (Spearman, natural train pairs):")
    cm = t[[f"dom_{k}" for k in CORE]].corr(method="spearman").round(2)
    print(cm.to_string())

    results = {"easy_features": easy_cols, "domains": DOMAINS, "diseases": int(len(d)), "external_check": ext, "models": {}}
    fitted = {}
    for kind in ("ridge", "gradient_boosting"):
        m = fit(kind, data["train"][easy_cols].values, y["train"], w); fitted[kind] = m
        res = {}
        for s in ("validation", "test"):
            p = np.clip(m.predict(data[s][easy_cols].values), 0, 1)
            res[s] = {"natural": metrics(y[s][rnd[s]], p[rnd[s]]), "informative": metrics(y[s][~rnd[s]], p[~rnd[s]]), "all": metrics(y[s], p)}
        res["retrieval_test"] = retrieval(d, rows, gt, m, easy_cols)
        results["models"][kind] = res
    sims = [c for c in easy_cols if c.endswith("_sim")]
    b = metrics(y["test"][rnd["test"]], data["test"][sims].mean(axis=1).values[rnd["test"]])
    results["baselines"] = {"train_mean_natural_test": metrics(y["test"][rnd["test"]], np.full(rnd["test"].sum(), np.average(y["train"], weights=w))),
                            "easy_average_natural_test_rank_only": {"spearman": b["spearman"]}}

    # ---- sensitivity of conclusions to the ground-truth definition
    pc = np.linalg.eigh(np.cov(t[[f"dom_{k}" for k in CORE]].values.T))[1][:, -1]
    pc = np.abs(pc) / np.abs(pc).sum()
    variants = {"equal (default)": lambda df: df.target_similarity.values,
                "phenotype-heavy 2:1:1": lambda df: composite(df, {"phenotype": 2, "genetic": 1, "mechanism": 1}),
                "genetic-heavy 1:2:1": lambda df: composite(df, {"phenotype": 1, "genetic": 2, "mechanism": 1}),
                "mechanism-heavy 1:1:2": lambda df: composite(df, {"phenotype": 1, "genetic": 1, "mechanism": 2}),
                "PC1 weights " + str([round(float(x), 2) for x in pc]): lambda df: composite(df, dict(zip(CORE, pc))),
                "raw IC-Jaccard, equal (no percentile)": lambda df: composite(df.assign(**{f"dom_{k}": np.nanmean(np.vstack([df[f"raw_{f}"].where(df[f"pct_{f}"].notna()) for f in DOMAINS[k]]), axis=0) for k in DOMAINS}), EQUAL)}
    sens = {}
    base_test = variants["equal (default)"](data["test"])[rnd["test"]]
    for name, fn in variants.items():
        yt = {s: fn(df) for s, df in data.items()}
        m = fit("gradient_boosting", data["train"][easy_cols].values, yt["train"], w)
        p = m.predict(data["test"][easy_cols].values)
        mm = metrics(yt["test"][rnd["test"]], p[rnd["test"]])
        sens[name] = {"corr_with_default_target": float(spearmanr(base_test, yt["test"][rnd["test"]])[0]), "model_r2": mm["r2"], "model_spearman": mm["spearman"]}
    results["sensitivity"] = sens

    # ---- ablation: drop one easy-feature GROUP at a time (gradient boosting, natural test pairs)
    ab, full = {}, results["models"]["gradient_boosting"]["test"]["natural"]
    for grp in sorted(set(groups[c] for c in easy_cols)):
        keep = [c for c in easy_cols if groups[c] != grp]
        m = fit("gradient_boosting", data["train"][keep].values, y["train"], w)
        mm = metrics(y["test"][rnd["test"]], np.clip(m.predict(data["test"][keep].values), 0, 1)[rnd["test"]])
        ab[grp] = {"r2_without": mm["r2"], "spearman_without": mm["spearman"], "r2_drop": full["r2"] - mm["r2"], "spearman_drop": full["spearman"] - mm["spearman"]}
    results["ablation_drop_group"] = ab
    json.dump(results, open(OUT / "results_groundtruth_v2.json", "w"), indent=2)

    for kind, r in results["models"].items():
        print(f"\n[{kind}]")
        for s in ("validation", "test"):
            for k in ("natural", "informative", "all"):
                m = r[s][k]
                print(f"  {s:10} {k:11} n={m['n']:5}  R2={m['r2']:6.3f}  RMSE={m['rmse']:.3f}  Spearman={m['spearman']:.3f}")
        print("  retrieval (test diseases ranked against each other):", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r["retrieval_test"].items()})
    print("\nbaselines:", {k: {m: round(v, 3) for m, v in bb.items() if m != "n"} for k, bb in results["baselines"].items()})
    print("\nSENSITIVITY to ground-truth definition (gradient boosting on natural test pairs):")
    for k, v in sens.items():
        print(f"  {k:48} target corr with default {v['corr_with_default_target']:.3f} | model R2 {v['model_r2']:.3f}  Spearman {v['model_spearman']:.3f}")
    print("\nABLATION (drop one easy-feature group):")
    for grp, v in sorted(ab.items(), key=lambda kv: -kv[1]["spearman_drop"]):
        print(f"  {grp:15} without it: R2={v['r2_without']:.3f} Spearman={v['spearman_without']:.3f}   (drop R2 {v['r2_drop']:+.3f}, Spearman {v['spearman_drop']:+.3f})")


if __name__ == "__main__":
    sys.exit(main())
