"""Feature-stratified pair candidates (PAPER_SCRAPING_METHOD.md steps 1-2), from .data/features.

Universe: Orphanet 'Disorder' entries of type Disease / Malformation syndrome with HPO or gene features
(groups, categories and subtypes excluded). Strata A-E as in the method; F (random cross-class) is
drawn from diseases sharing nothing. Candidates are oversampled; probe_pairs.py then keeps pairs that
actually have co-mention literature. Seeded, so reproducible."""
import random, re, collections as C
import pandas as pd
from pathlib import Path
R=Path(__file__).resolve().parent
SEED=20261004; N_CAND=int(__import__('sys').argv[1]) if len(__import__('sys').argv)>1 else 900
d=pd.read_parquet(R.parent.parent/'.data/features/diseases.parquet')
u=d[(d.disorder_group=='Disorder')&d.disorder_type.isin(['Disease','Malformation syndrome'])]
u=u[(u.n_hpo_phenotypes>=5)|(u.n_genes>0)]
u=u[u.name.str.len().between(6,70)]
used=set()
old=R/'literature_disease_pairs_pairfirst.csv'
rec={r.orpha_id:r for r in u.itertuples()}
ids=sorted(rec)
S=lambda x:set(x) if x is not None else set()
ph={i:S(rec[i].hpo_ids) for i in ids}; gn={i:S(rec[i].gene_symbols) for i in ids}
dr={i:S(rec[i].approved_drug_ids) for i in ids}
par={i:(rec[i].preferential_parent_ids[0] if len(rec[i].preferential_parent_ids) else None) for i in ids}
bs={i:S(rec[i].body_system_names)-{'Rare genetic disease','Rare developmental defect during embryogenesis'} for i in ids}
W=lambda s:set(re.findall(r'[a-z0-9]+',s.lower()))-{'syndrome','disease','type','of','and','with','the','a','due','to','deficiency'}
def subtype_like(a,b):
    A,B=W(rec[a].name),W(rec[b].name)
    return bool(A and B) and len(A&B)/len(A|B)>=0.4
def jac(a,b):
    A,B=ph[a],ph[b]; return len(A&B)/len(A|B) if A and B else 0.0
def inv(m,cap):
    x=C.defaultdict(list)
    for i,s in m.items():
        for t in s: x[t].append(i)
    return {t:v for t,v in x.items() if len(v)<=cap}
pairs={}
def add(a,b,tag):
    if a==b: return
    k=tuple(sorted((a,b)))
    if k not in pairs: pairs[k]=tag
cand=C.defaultdict(set)
# phenotype neighbours
for t,v in inv(ph,250).items():
    pass
cnt=C.defaultdict(C.Counter)
for t,v in inv(ph,250).items():
    for x in v:
        for y in v:
            if x<y: cnt[x][y]+=1
P={}
for x,c in cnt.items():
    for y,n in c.items():
        if n>=4: P[(x,y)]=jac(x,y)
ginv=inv(gn,60); G=set()
for g,v in ginv.items():
    for i in range(len(v)):
        for j in range(i+1,len(v)): G.add(tuple(sorted((v[i],v[j]))))
dinv=inv(dr,60); D=set()
for g,v in dinv.items():
    for i in range(len(v)):
        for j in range(i+1,len(v)): D.add(tuple(sorted((v[i],v[j]))))
strata=C.defaultdict(list)
for (a,b) in set(P)|G|D:
    if subtype_like(a,b): continue
    j=P.get((a,b)) if (a,b) in P else jac(a,b)
    sg=gn[a]&gn[b]; sd=dr[a]&dr[b]
    if j>=0.3 and sg: s='A_close_relatives'
    elif j>=0.3 and gn[a] and gn[b] and not sg: s='B_phenotype_twins_diff_genes'
    elif sd and not (bs[a]&bs[b]): s='D_shared_treatment_diff_systems'
    elif sg and j<0.15: s='C_genetic_cousins'
    elif sd: s='D_shared_treatment_diff_systems'
    else: continue
    strata[s].append((a,b,j,len(sg),len(sd)))
# E: same classification parent, nothing else shared
rng=random.Random(SEED)
bypar=C.defaultdict(list)
for i in ids:
    if par[i]: bypar[par[i]].append(i)
E=[]
for p,v in bypar.items():
    if 3<=len(v)<=400:
        for _ in range(min(120,len(v)*3)):
            a,b=rng.sample(v,2); k=tuple(sorted((a,b)))
            if gn[a]&gn[b] or dr[a]&dr[b] or jac(a,b)>0.05 or subtype_like(a,b): continue
            E.append((k[0],k[1],jac(a,b),0,0))
strata['E_same_class_unrelated_biology']=list(set(E))
F=[]
while len(F)<4000:
    a,b=rng.sample(ids,2)
    if par[a]==par[b] or bs[a]&bs[b] or gn[a]&gn[b] or dr[a]&dr[b] or jac(a,b)>0 or subtype_like(a,b): continue
    F.append((*sorted((a,b)),0.0,0,0))
strata['F_random_cross_class']=list(set(F))
out=[]; deg=C.Counter(); cap=4
for s in sorted(strata):
    L=sorted(strata[s]); rng.shuffle(L); n=0
    for a,b,j,sg,sd in L:
        if deg[a]>=cap or deg[b]>=cap: continue
        deg[a]+=1; deg[b]+=1; n+=1
        out.append(dict(pair_id=f'{a}|{b}',disease_a=rec[a].name,disease_b=rec[b].name,disease_a_orpha_id=a,disease_b_orpha_id=b,
          synonyms_a='|'.join(rec[a].synonyms[:6]),synonyms_b='|'.join(rec[b].synonyms[:6]),stratum=s,pheno_jaccard=f'{j:.2f}',
          n_shared_genes=sg,n_shared_drugs=sd,
          status_a='rare' if rec[a].prevalence_class not in ('>1 / 1000',) else 'common',status_b='rare' if rec[b].prevalence_class not in ('>1 / 1000',) else 'common'))
        if n>=N_CAND: break
    print(s,len(strata[s]),'available ->',n)
pd.DataFrame(out).to_csv(R/'pairs_candidates_v2.csv',index=False)
print(len(out),'candidates')
