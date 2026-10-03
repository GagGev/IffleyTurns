"""Rebuild edge_ranking.csv from literature_disease_pairs.csv.
Evidence score per paper = study design (0-8) + sample size (0-3) + molecular dimension (0/1) + full text read (0/1) - 2 if 'unrelated'.
Edge score = best paper score + replication bonus (extra independent papers, max 3). weight = score / 15.
similarity_score (edge) = evidence-weighted mean of the per-paper similarity_score."""
import csv,re,collections as C
src='literature_review/literature_disease_pairs.csv'
rows=list(csv.DictReader(open(src)))
def design(s):
    s=s.lower()
    if 'case report' in s: return 0,'case report'
    if 'meta-analysis' in s or 'multicentre' in s or 'multicenter' in s or 'comparative' in s or 'cohort' in s or 'retrospective' in s: return 4,'comparative/cohort/multicentre study'
    if 'letter' in s or s=='': return 1,'letter/unspecified design'
    if 'case series' in s or 'family' in s: return 1,'case series/family study'
    if 'review' in s: return 2,'narrative review (no new data on this pair)'
    if 'original' in s or 'gwas' in s or 'preclinical' in s: return 3,'original research'
    return 4,'comparative/cohort/multicentre study'
def nsize(s):
    m=re.findall(r'\d[\d,]*',s or '');return max([int(x.replace(',','')) for x in m],default=0)
def score(x):
    d,dn=design(x['study_type']);n=nsize(x['sample_size']);sc=d*2
    sc+=0 if n==0 else (1 if n<20 else 2 if n<100 else 3)
    sc+=1 if x['similarity_dimension'].split(';')[0].strip() in('genes','pathways','treatment') else 0
    sc+=1 if 'full text read' in x['full_text_access'] else 0
    sc-=2 if x['relationship']=='unrelated' else 0
    return sc,dn,n
E_=C.defaultdict(list)
for x in rows:
    a,b=x['disease_a_canonical'].strip(),x['disease_b_canonical'].strip()
    if a and b and a!=b: E_[frozenset((a,b))].append(x)
out=[]
for k,xs in E_.items():
    sc=[score(x) for x in xs];papers={x['link'] for x in xs}
    total=max(s[0] for s in sc)+min(len(papers)-1,3)
    out.append((total,sorted(k),xs,sc,len(papers)))
out.sort(key=lambda t:(-t[0],t[1]))
H=['rank','score','weight','similarity_score','disease_a','disease_b','n_papers','best_study_design','relationship','dimension','top_paper','year','link','weakness_flags']
w=csv.writer(open('literature_review/edge_ranking.csv','w',newline=''));w.writerow(H)
for i,(t,k,xs,sc,n) in enumerate(out,1):
    j=max(range(len(xs)),key=lambda q:sc[q][0]);x=xs[j];fl=[]
    if n==1: fl.append('single paper')
    if sc[j][1]=='case report': fl.append('case report only')
    if sc[j][2]==0: fl.append('no sample size recorded')
    if re.search('mimic|misdiagnos',x['finding'],re.I): fl.append('mimic/misdiagnosis, not mechanistic relatedness')
    if all(design(y['study_type'])[0]<=1 for y in xs): fl.append('no study above case-series level')
    ws=[max(s[0],0)+1 for s in sc]
    sim=sum(a*float(y['similarity_score'] or 0) for a,y in zip(ws,xs))/sum(ws)
    w.writerow([i,t,f"{max(t,0)/15:.3f}",f"{sim:.2f}",k[0],k[1],n,sc[j][1],x['relationship'],x['similarity_dimension'],x['paper_title'],x['year'],x['link'],'; '.join(fl)])
print(len(out),'edges')
