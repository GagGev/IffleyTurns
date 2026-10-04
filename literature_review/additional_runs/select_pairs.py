"""Pair-first selection (PAPER_SCRAPING_METHOD.md steps 1-2), lightweight version.

Uses the HPO annotation files (ORPHA diseases with phenotypes and genes) because the
full feature tables were not built. Strata A/B/C are computed from HPO-phenotype and
gene overlap; D (shared drug) and E/F (negatives) need the feature tables.
Deterministic seed; each disease capped at ~5% of pairs.
"""
import csv, random, collections as C
from pathlib import Path
R=Path(__file__).resolve().parent; H=R.parent.parent/'.data/databases/hpo'
SEED=20261004; PER_STRATUM=80
pheno=C.defaultdict(set); name={}
for l in open(H/'phenotype.hpoa'):
    if l.startswith('#') or l.startswith('database_id'): continue
    f=l.rstrip('\n').split('\t')
    if f[0].startswith('ORPHA:') and f[2]!='NOT' and f[10]=='P':
        pheno[f[0]].add(f[3]); name[f[0]]=f[1]
genes=C.defaultdict(set)
for l in open(H/'genes_to_disease.txt'):
    f=l.rstrip('\n').split('\t')
    if f[3].startswith('ORPHA:'): genes[f[3]].add(f[1])
mapped={r['orpha_id'] for r in csv.DictReader(open(R.parent/'disease_name_mapping.csv'))}
used=set()
for r in csv.DictReader(open(R.parent/'literature_disease_pairs.csv')):
    used.add(frozenset((r['disease_a_orpha_id'],r['disease_b_orpha_id'])))
ids=sorted(d for d in pheno if len(pheno[d])>=8 and 'ORPHA' in d)
# exclude group/umbrella entries by name heuristic
bad=('syndrome, ','susceptibility','particular clinical','rare ','familial or sporadic')
ids=[d for d in ids if not any(b in name[d].lower() for b in bad) or d in mapped]
byg=C.defaultdict(list)
for d in ids:
    for g in genes.get(d,()): byg[g].add(d) if False else byg[g].append(d)
cand={}
def jac(a,b): return len(a&b)/len(a|b)
# phenotype-near neighbours via inverted index on HPO terms
inv=C.defaultdict(list)
for d in ids:
    for t in pheno[d]: inv[t].append(d)
for d in ids:
    cnt=C.Counter()
    for t in pheno[d]:
        if len(inv[t])<300:
            for e in inv[t]:
                if e>d: cnt[e]+=1
    for e,c in cnt.items():
        if c>=4:
            cand[(d,e)]=jac(pheno[d],pheno[e])
for g,ds in byg.items():
    for i in range(len(ds)):
        for j in range(i+1,len(ds)):
            k=tuple(sorted((ds[i],ds[j]))); cand.setdefault(k,jac(pheno[k[0]],pheno[k[1]]))
strata=C.defaultdict(list)
for (a,b),p in cand.items():
    if a==b or frozenset((a,b)) in used: continue
    sg=genes[a]&genes[b]; ga,gb=bool(genes[a]),bool(genes[b])
    if p>=0.35 and sg: s='A_close_relatives'
    elif p>=0.35 and ga and gb and not sg: s='B_phenotype_twins_diff_genes'
    elif sg and p<0.2: s='C_genetic_cousins'
    else: continue
    strata[s].append((a,b,p,len(sg)))
rng=random.Random(SEED); out=[]; deg=C.Counter(); cap=max(3,int(0.05*PER_STRATUM*3))
for s in sorted(strata):
    L=sorted(strata[s]); rng.shuffle(L); n=0
    for a,b,p,sg in L:
        if deg[a]>=cap or deg[b]>=cap: continue
        deg[a]+=1; deg[b]+=1; n+=1
        out.append(dict(pair_id=f'{a}|{b}',disease_a=name[a],disease_b=name[b],disease_a_orpha_id=a,disease_b_orpha_id=b,
            synonyms_a='',synonyms_b='',stratum=s,pheno_jaccard=f'{p:.2f}',n_shared_genes=sg,status_a='rare',status_b='rare'))
        if n>=PER_STRATUM: break
w=csv.DictWriter(open(R/'pairs_selected.csv','w',newline=''),fieldnames=list(out[0]));w.writeheader();w.writerows(out)
print(len(ids),'diseases;',{s:len(v) for s,v in strata.items()},'->',len(out),'pairs selected')
