"""Cheap Europe PMC hit-count probe for pairs_candidates_v2.csv (title/abstract co-mention of both diseases)."""
import csv, sys, time, requests, concurrent.futures as cf
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from search_pair_papers import pair_names, clause, API
R=Path(__file__).resolve().parent
rows=list(csv.DictReader(open(R/'pairs_candidates_v2.csv')))
local=__import__('threading').local()
def probe(p):
    s=getattr(local,'s',None) or requests.Session(); local.s=s
    names=pair_names(p)
    if names is None: return -2
    na,nb=names
    q=f"{clause(na)} AND {clause(nb)} AND HAS_ABSTRACT:y AND LANG:eng"
    for k in range(4):
        try:
            r=s.get(API,params={'query':q,'resultType':'lite','format':'json','pageSize':1},timeout=60); r.raise_for_status()
            return int(r.json()['hitCount'])
        except Exception: time.sleep(2*(k+1))
    return -1
with cf.ThreadPoolExecutor(8) as ex: hits=list(ex.map(probe,rows))
for p,h in zip(rows,hits): p['hit_count']=h
w=csv.DictWriter(open(R/'pairs_probed_v2.csv','w',newline=''),fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
import collections as C
for s in sorted({p['stratum'] for p in rows}):
    h=[p['hit_count'] for p in rows if p['stratum']==s]
    print(s,len(h),'hit>=1:',sum(x>=1 for x in h),'hit>=3:',sum(x>=3 for x in h),'failed:',sum(x<0 for x in h))
