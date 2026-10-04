"""Test of a fix for the annotation-size confound found by audit.py.

Problem: IC-Jaccard and a *global* percentile make the target depend on how many annotations the two diseases have
(a model that sees only annotation counts beats the easy-feature model).  Fix tested here: convert each raw similarity
to a percentile within SIZE-MATCHED random pairs (bins of log annotation counts), so the target asks "is this overlap
unusual for two diseases with this many annotations?".

  python easy_hard/audit_size_fix.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

sys.argv = sys.argv[:1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit  # noqa: E402
import run  # noqa: E402

warnings.filterwarnings("ignore")
N_BINS = 10


class SizeAwareGT(run.GroundTruth):
    def __init__(self, hf, bg_i, bg_j):
        self.hf = hf
        self.cnt = {f: np.asarray(m.sum(axis=1)).ravel() for f, m in hf.mats.items()}
        self.bg, self.edges = {}, {}
        for f in hf.mats:
            r = hf.raw(f, bg_i, bg_j)
            s = self._size(f, bg_i, bg_j)
            edges = np.quantile(s, np.linspace(0, 1, N_BINS + 1)[1:-1])
            b = np.searchsorted(edges, s)
            self.edges[f] = edges
            self.bg[f] = [np.sort(r[b == k]) for k in range(N_BINS)]

    def _size(self, f, i, j):
        return np.log1p(self.cnt[f][i]) + np.log1p(self.cnt[f][j])

    def score(self, i, j):
        out = {}
        for dom, fs in run.DOMAINS.items():
            pcts = []
            for f in fs:
                r = self.hf.raw(f, i, j)
                b = np.searchsorted(self.edges[f], self._size(f, i, j))
                p = np.array([np.searchsorted(self.bg[f][k], x, side="left") / max(len(self.bg[f][k]), 1) for k, x in zip(b, r)])
                avail = self.hf.has[f][i] & self.hf.has[f][j]
                out[f"raw_{f}"], out[f"pct_{f}"] = r, np.where(avail, p, np.nan)
                pcts.append(out[f"pct_{f}"])
            out[f"dom_{dom}"] = np.nanmean(np.vstack(pcts), axis=0) if len(pcts) > 1 else pcts[0]
        return pd.DataFrame(out)


def evaluate(d, gt_cls, label):
    rows = list(d.itertuples(index=False))
    hf = run.HardFeatures(d)
    rng = np.random.default_rng(run.SEED)
    tr = np.where(d.split.values == "train")[0]
    pairs = rng.choice(tr, (run.N_BACKGROUND, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    gt = gt_cls(hf, pairs.min(axis=1), pairs.max(axis=1))
    data = {s: run.sample_pairs(d, rows, s, n, rng, gt) for s, n in {"train": 40000, "validation": 4000, "test": 8000}.items()}
    groups = run.easy_groups()
    easy = [c for c in data["train"].columns if c in groups]
    w = data["train"].w.values
    cnt = np.log1p(np.vstack([np.asarray(m.sum(axis=1)).ravel() for m in hf.mats.values()]).T)

    def counts(df):
        gi, gj = df.gi.values, df.gj.values
        return np.c_[cnt[gi] + cnt[gj], np.abs(cnt[gi] - cnt[gj])]
    te = data["test"]; r = (te.kind == "random").values
    y = te.target_similarity.values[r]
    hgb = lambda: HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05, random_state=run.SEED)
    m_easy = hgb().fit(data["train"][easy].values, data["train"].target_similarity.values, sample_weight=w)
    m_cnt = hgb().fit(counts(data["train"]), data["train"].target_similarity.values, sample_weight=w)
    comb = lambda df: np.c_[df[easy].values, counts(df)]
    m_comb = hgb().fit(comb(data["train"]), data["train"].target_similarity.values, sample_weight=w)
    res = {"label": label, "diseases": len(d),
           "easy_plus_counts_model": run.metrics(y, np.clip(m_comb.predict(comb(te)), 0, 1)[r]),
           "easy_model": run.metrics(y, np.clip(m_easy.predict(te[easy].values), 0, 1)[r]),
           "count_only_model": run.metrics(y, np.clip(m_cnt.predict(counts(te)), 0, 1)[r])}
    # literature agreement
    papers = audit.load_paper_pairs()
    idx = {o: i for i, o in enumerate(d.orpha_id)}
    use = [(p, e) for p, e in papers.items() if all(o in idx for o in p) and len(p) == 2]
    gi = np.array([idx[sorted(p)[0]] for p, _ in use]); gj = np.array([idx[sorted(p)[1]] for p, _ in use])
    sc = run.pair_frame(rows, gi, gj, gt)
    lit = pd.DataFrame([e for _, e in use])
    res["literature_pairs"] = len(use)
    res["literature_rho_composite"] = float(spearmanr(sc.target_similarity, lit.paper_sim)[0])
    for dom, pdim in (("dom_phenotype", "paper_phenotype"), ("dom_genetic", "paper_genetic"), ("dom_mechanism", "paper_mechanism")):
        ok = sc[dom].notna().values & lit[pdim].notna().values
        res[f"literature_rho_{dom}"] = float(spearmanr(sc[dom].values[ok], lit[pdim].values[ok])[0])
    res["literature_rho_easy_model"] = float(spearmanr(np.clip(m_easy.predict(sc[easy].values), 0, 1), lit.paper_sim)[0])
    return res


def main():
    d = run.load_diseases(clean_hpo=True, curated_genes=True)  # the cleaner universe (audit experiment E2)
    out = [evaluate(d, run.GroundTruth, "size-agnostic percentile (as shipped), clean universe"),
           evaluate(d, SizeAwareGT, "size-matched percentile (proposed fix), clean universe")]
    for r in out:
        print(f"\n{r['label']}  ({r['diseases']} diseases, {r['literature_pairs']} literature pairs)")
        print(f"  easy-feature model        : R2={r['easy_model']['r2']:.3f}  Spearman={r['easy_model']['spearman']:.3f}")
        print(f"  annotation-count-only     : R2={r['count_only_model']['r2']:.3f}  Spearman={r['count_only_model']['spearman']:.3f}")
        print(f"  easy features + counts    : R2={r['easy_plus_counts_model']['r2']:.3f}  Spearman={r['easy_plus_counts_model']['spearman']:.3f}")
        print(f"  literature agreement      : composite rho={r['literature_rho_composite']:.3f} | phenotype {r['literature_rho_dom_phenotype']:.3f} "
              f"| genetic {r['literature_rho_dom_genetic']:.3f} | mechanism {r['literature_rho_dom_mechanism']:.3f} | easy-model prediction rho={r['literature_rho_easy_model']:.3f}")
    json.dump(out, open(run.OUT / "audit_size_fix.json", "w"), indent=2)


if __name__ == "__main__":
    main()
