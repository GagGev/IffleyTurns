"""Assemble the pair-first paper dataset (PAPER_SCRAPING_METHOD.md step 6) from the scored candidate papers.

Sources (each: pairs file, candidate_papers jsonl, manual score table):
  pairfirst-1  pairs_selected.csv     candidate_papers.jsonl     scores_pairfirst.py     (HPO-based pair selection)
  pairfirst-2  pairs_selected_v2.csv  candidate_papers_v2.jsonl  scores_pairfirst_v2.py  (feature-stratified selection)
Output layout matches literature_disease_pairs.csv / paper_dimension_scores.csv plus stratum, query_type, low_evidence,
so splits.py can consume them. Scores are abstract-level and manual; weights sum to 1 per paper."""
import csv, json, collections as C, importlib.util, re
from pathlib import Path
R=Path(__file__).resolve().parent
def load(mod):
    s=importlib.util.spec_from_file_location('m',R/mod); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m.S
DIMS={'P':'phenotype','G':'genetic','M':'mechanism','T':'therapeutic','N':'natural_history','D':'diagnostic_confusability','C':'comorbidity'}
DIM_LABEL={'P':'phenotype','G':'genes','M':'pathways','T':'treatment','N':'epidemiology','D':'other (diagnostic)','C':'other (comorbidity)'}
SOURCES=[('pairfirst-1','pairs_selected.csv','candidate_papers.jsonl','scores_pairfirst.py',True),
         ('pairfirst-2','pairs_selected_v2.csv','candidate_papers_v2.jsonl','scores_pairfirst_v2.py',False)]
def clean(n): return re.sub(r'^(NON RARE IN EUROPE|OBSOLETE):\s*','',n)
def status(n): return 'common' if n.startswith('NON RARE IN EUROPE') else 'rare'
def stype(t):
    t=' '.join(t).lower()
    if 'case report' in t: return 'case report'
    if 'review' in t or 'guideline' in t or 'consensus' in t: return 'review'
    if 'comparative' in t or 'multicenter' in t: return 'comparative study'
    return 'original research'
recs=[]; seen=set()
for batch,pf,cf,sf,drop_hemo in SOURCES:
    pairs={p['pair_id']:p for p in csv.DictReader(open(R/pf))}; S=load(sf)
    rows=[json.loads(l) for l in open(R/cf)]
    if drop_hemo: rows=[r for r in rows if r['disease_a']!='Severe hemophilia A']
    for i,r in enumerate(rows): r['idx']=i
    for r in sorted((r for r in rows if r['idx'] in S),key=lambda r:(r['pair_id'],r['idx'])):
        key=(r['pair_id'],r['pmid'])
        if key in seen or r['disease_a'].startswith('OBSOLETE') or r['disease_b'].startswith('OBSOLETE'): continue
        seen.add(key); finding,d=S[r['idx']]; recs.append((batch,pairs[r['pair_id']],r,finding,d))
npair=C.Counter(p['pair_id'] for _,p,_,_,_ in recs)
H=['disease_a','disease_b','disease_a_canonical','disease_b_canonical','disease_a_orpha_id','disease_b_orpha_id','disease_a_orpha_mapping_status','disease_b_orpha_mapping_status','status_a','status_b','similarity_dimension','relationship','similarity_score','finding','key_details','study_type','sample_size','paper_title','year','link','pmid','open_access','full_text_access','evidence_checked','batch','stratum','query_type','low_evidence']
DH=['row_id','pmid','disease_a','disease_b','paper_title','year','link']+[f'w_{d}' for d in DIMS.values()]+[f's_{d}' for d in DIMS.values()]+['notes','evidence']
fw=csv.DictWriter(open(R/'literature_disease_pairs_pairfirst.csv','w',newline=''),fieldnames=H); fw.writeheader()
dw=csv.DictWriter(open(R/'paper_dimension_scores_pairfirst.csv','w',newline=''),fieldnames=DH); dw.writeheader()
for n,(batch,p,r,finding,d) in enumerate(recs,1):
    tw=sum(w for w,_ in d.values()); assert abs(tw-1)<1e-6,(batch,r['pmid'],tw)
    sim=sum(w*s for w,s in d.values())/tw
    dims='; '.join(DIM_LABEL[k] for k in sorted(d,key=lambda k:-d[k][0])[:2])
    rel='similar' if sim>=0.65 else ('related but distinct' if sim>=0.2 else 'unrelated')
    link=f"https://europepmc.org/article/PMC/{r['pmcid']}" if r['pmcid'] else (f"https://doi.org/{r['doi']}" if r['doi'] else f"https://europepmc.org/article/MED/{r['pmid']}")
    oa='yes' if r['open_access']=='Y' else 'no'
    a,b=p['disease_a'],p['disease_b']
    fw.writerow(dict(disease_a=clean(a),disease_b=clean(b),disease_a_canonical=clean(a),disease_b_canonical=clean(b),
        disease_a_orpha_id=p['disease_a_orpha_id'],disease_b_orpha_id=p['disease_b_orpha_id'],disease_a_orpha_mapping_status='feature_table',disease_b_orpha_mapping_status='feature_table',
        status_a=status(a),status_b=status(b),similarity_dimension=dims,relationship=rel,similarity_score=f'{sim:.2f}',finding=finding,key_details='',
        study_type=stype(r['pub_types']),sample_size='',paper_title=r['title'],year=r['year'],link=link,pmid=r['pmid'],open_access=oa,
        full_text_access='yes - free full text in PMC/Europe PMC (not read)' if oa=='yes' else 'no - not marked open access in Europe PMC',
        evidence_checked='abstract only',batch=batch,stratum=p['stratum'],query_type=r['query_type'],low_evidence='yes' if npair[p['pair_id']]<3 else 'no'))
    row=dict(row_id=n,pmid=r['pmid'],disease_a=clean(a),disease_b=clean(b),paper_title=r['title'],year=r['year'],link=link,notes=finding,evidence='abstract only')
    for k,name in DIMS.items():
        w,s=d.get(k,(0,None)); row[f'w_{name}']=f'{w:.2f}'; row[f's_{name}']='' if w==0 else s
    dw.writerow(row)
print(len(recs),'rows;',len(npair),'pairs; papers/pair',sorted(C.Counter(npair.values()).items()))
print('by stratum (pairs)',C.Counter(next(p['stratum'] for _,p,_,_,_ in recs if p['pair_id']==k) for k in npair))
print('by stratum (rows)',C.Counter(p['stratum'] for _,p,_,_,_ in recs))
