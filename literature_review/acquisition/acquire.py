"""Reproducible, review-gated Europe PMC calibration corpus (standard library only)."""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import random
import re
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / '.data/literature_acquisition/v1'
API = 'https://www.ebi.ac.uk/europepmc/webservices/rest'
VERSION = 'rare-disease-calibration-1.0'
DIMENSIONS = {
    'phenotype': 'phenotyp* OR "clinical features" OR manifestation* OR symptom*',
    'genetic': 'gene* OR variant* OR inherit* OR genotype*',
    'mechanism': 'pathway* OR mechanis* OR molecular OR pathogenesis',
    'therapeutic': 'treatment* OR therap* OR drug* OR intervention*',
    'natural_history': '"natural history" OR progression OR onset OR survival',
    'diagnostic_confusability': '"differential diagnosis" OR mimic* OR misdiagnos* OR distinguish*',
    'comorbidity': 'comorbid* OR "co-occur" OR association OR "risk of"',
}
CONTROL = ('distinguish* OR "differential diagnosis" OR unrelated OR distinct OR '
           '"no association" OR "negative association" OR reclassification OR '
           '"non-replication" OR "different mechanisms"')


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(value):
    if isinstance(value, str):
        value = value.encode('utf-8')
    return hashlib.sha256(value).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def normalize(text):
    return re.sub(r'\s+', ' ', text.casefold()).strip()


def plain(element):
    return re.sub(r'\s+', ' ', ''.join(element.itertext())).strip()


def dictionary():
    path = ROOT / '.data/databases/orphadata/en_product1.xml'
    root = ET.parse(path).getroot()
    entities, names = {}, collections.defaultdict(set)
    for node in root.findall('DisorderList/Disorder'):
        oid = 'ORPHA:' + node.findtext('OrphaCode')
        synonyms = [s.text for s in node.findall('SynonymList/Synonym') if s.text]
        name = node.findtext('Name')
        external = []
        for ref in node.findall('ExternalReferenceList/ExternalReference'):
            external.append({'source': ref.findtext('Source'), 'id': ref.findtext('Reference'),
                             'relation': ref.findtext('DisorderMappingRelation/Name'),
                             'validation': ref.findtext('DisorderMappingValidationStatus/Name')})
        entities[oid] = {'orpha_id': oid, 'preferred_name': name, 'synonyms': synonyms,
                         'entity_type': node.findtext('DisorderType/Name'),
                         'external_identifiers': external, 'mapping_confidence': 'exact_source',
                         'rarity_status': 'requires_prevalence_review',
                         'source': 'https://www.orphadata.com/data/xml/en_product1.xml'}
        for term in [name] + synonyms:
            names[normalize(term)].add(oid)
    return entities, names, {'date': root.attrib.get('date'), 'version': root.attrib.get('version'),
                             'sha256': digest(path.read_bytes()), 'license': 'CC-BY-4.0'}


def register(out):
    if (out / 'protocol.json').exists():
        raise ValueError('Protocol already registered; use another output directory for a new version.')
    entities, names, release = dictionary()
    seed_path = ROOT / 'literature_review/literature_disease_pairs.csv'
    candidates, rejected = set(), collections.Counter()
    for row in csv.DictReader(seed_path.open(encoding='utf-8-sig')):
        a, b = names.get(normalize(row['disease_a']), set()), names.get(normalize(row['disease_b']), set())
        if len(a) != 1 or len(b) != 1:
            rejected['ambiguous_or_unmapped_name'] += 1
            continue
        pair = tuple(sorted([next(iter(a)), next(iter(b))]))
        types = [entities[x]['entity_type'] for x in pair]
        if pair[0] == pair[1] or types[0] != types[1] or types[0] not in {
                'Disease', 'Malformation syndrome', 'Clinical subtype', 'Etiological subtype'}:
            rejected['incompatible_or_broad_granularity'] += 1
            continue
        candidates.add(pair)
    selected, degree = [], collections.Counter()
    for pair in sorted(candidates, key=lambda x: digest('|'.join(x))):
        if any(degree[x] >= 2 for x in pair):
            continue
        selected.append(pair)
        degree.update(pair)
        if len(selected) == 30:
            break
    assert len(selected) == 30
    diseases = [entities[x] for x in sorted(degree)]
    for disease in diseases:
        terms = [disease['preferred_name']] + disease['synonyms']
        disease['search_terms'] = [s for s in terms if len(s) >= 6 and not s.isupper()
                                   and len(names[normalize(s)]) == 1][:8]
        disease['excluded_search_terms'] = [s for s in terms if s not in disease['search_terms']]
        if not disease['search_terms']:
            raise ValueError('No unambiguous search terms: ' + disease['orpha_id'])
    rubric = ROOT / 'experiments/SCIENTIFIC_LITERATURE_ACQUISITION_METHOD.md'
    protocol = {
        'title': 'Rare-disease literature: prospective acquisition and rubric-calibration pilot',
        'version': VERSION, 'registered_at': now(), 'date_range': ['2000-01-01', '2026-10-04'],
        'orphanet_release': release, 'seed_file_sha256': digest(seed_path.read_bytes()),
        'rubric_sha256': digest(rubric.read_bytes()), 'rubric_version': 'protocol-section-10-v1',
        'phase': '1: acquisition for calibration; not a validated benchmark',
        'selection': '30 exact-name Orphanet pairs; same entity type; SHA256 ordering; degree <=2. '
                     'Existing CSV contributes names only, never scores, findings, or votes.',
        'seed_selection_rejections': dict(rejected),
        'diseases': diseases, 'pairs': selected,
        'sources': [{'name': 'Europe PMC REST', 'api_version': '6.9', 'base_url': API}],
        'source_scope': 'Europe PMC includes PubMed records. Independent E-utilities, Crossref and '
                        'Unpaywall searches are deferred to benchmark expansion; no fabricated contact email.',
        'query_templates': {'A': 'pair unqualified, then each of seven dimensions',
                            'B': 'each disease AND union of dimension terms', 'C': 'pair AND control terms'},
        'dimension_terms': DIMENSIONS, 'control_terms': CONTROL,
        'limits': {'results_per_query': 20, 'full_text_attempts': 180, 'citation_seeds': 8,
                   'citation_rounds': 2, 'citation_page_size': 25},
        'stopping': 'All templates run. Capped queries/citation pages are explicitly incomplete; '
                    'budget exhaustion is NOT evidence saturation or a systematic-review stopping claim.',
        'inclusion': 'Both exact canonical entities; substantive dimension evidence; identifiable population '
                     'and model; compatible granularity. Human screening is required.',
        'exclusions': ['ambiguous_mapping', 'granularity_mismatch', 'cooccurrence_only',
                       'healthy_comparator', 'duplicate_family', 'retracted', 'no_direct_evidence',
                       'animal_as_clinical', 'case_report_as_population_comorbidity'],
        'license_policy': 'Only explicit CC-BY or CC0 full-text body enters reusable text exports. '
                          'NC, ND, SA, unspecified licenses require separate review. Metadata/abstracts '
                          'never automatically imply permission for text training.',
        'score_definitions': 'Exact section 10 of frozen rubric; ordinal 0..4, missing is null. '
                             'No automated cooccurrence-to-score conversion.',
        'primary_outcome': 'Audited dimension-specific ordinal evidence; no primary model claims in pilot.',
        'secondary_outcome': 'Licensed document corpus and locator-based review queue.',
        'controls': 'Search for explicit unrelated and related-but-distinct evidence; no absence-based negatives. '
                    'No unobserved control sampling until body system/prevalence/coverage matching is available.',
        'review': {'independent_reviewers': 2, 'minimum_screening_pilot': 100, 'kappa_threshold': 0.7,
                   'within_one_level_threshold': 0.9, 'test_extraction': 'two independent domain-informed reviewers',
                   'training_extraction': 'one extractor plus audit only after reliability threshold is met'},
        'aggregation': {'method': 'within-family median then weighted median',
                        'quality_weights': {'high': 4, 'moderate': 3, 'low': 2, 'very_low': 1},
                        'weight': 'quality_weight * extraction_confidence; cap each family at 12',
                        'sensitivity': 'unweighted family median; range and MAD'},
        'splits': {'paper_pair_family': 'connected components; deterministic 80/10/10 hash allocation',
                   'disease_held_out': '20% diseases hash-selected; mixed pairs quarantined; shared papers quarantined',
                   'time': 'train <=2022; validation 2023; test >=2024; crossing study families quarantined',
                   'status': 'provisional until family review and benchmark freeze'},
        'minimum_evidence': 'No primary claims below 100 held-out pairs/dimension and 50 examples/required group. '
                            'Primary labels need reviewed families, verified mapping and controls.',
        'sensitivity_analyses': ['full_text_only', 'high_moderate_quality', 'multiple_study_families',
                                  'paper_pair_family_blocked', 'disease_held_out', 'rare_rare_only',
                                  'compatible_granularity', 'unweighted_median', 'remove_each_hub'],
        'known_limitations': ['Pairs seeded from prior literature induce selection bias.',
                              'Orphanet membership does not establish prevalence <1/2000 in every population.',
                              'Candidate passages are retrieval aids, not screening decisions or evidence labels.',
                              'Study-family assignments begin unresolved; no primary supervised export until reviewed.'],
    }
    write_json(out / 'protocol.json', protocol)
    (out / 'rubric.md').write_bytes(rubric.read_bytes())
    (out / 'protocol.sha256').write_text(digest((out / 'protocol.json').read_bytes()), encoding='ascii')
    print(json.dumps({'registered': str(out / 'protocol.json'), 'pairs': len(selected), 'diseases': len(diseases)}))


SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS diseases(orpha_id TEXT PRIMARY KEY, preferred_name TEXT, entity_type TEXT, metadata_json TEXT);
CREATE TABLE IF NOT EXISTS pairs(pair_id TEXT PRIMARY KEY, orpha_id_a TEXT REFERENCES diseases, orpha_id_b TEXT REFERENCES diseases,
 control_type TEXT NOT NULL DEFAULT 'unknown', mapping_status TEXT NOT NULL DEFAULT 'candidate_exact_dictionary',
 CHECK(orpha_id_a < orpha_id_b), CHECK(control_type IN ('unknown','related','related_but_distinct','verified_unrelated','unobserved')));
CREATE TABLE IF NOT EXISTS requests(request_id TEXT PRIMARY KEY, endpoint TEXT, requested_at TEXT, completed_at TEXT,
 status INTEGER, response_sha256 TEXT, raw_path TEXT, error TEXT, attempts INTEGER);
CREATE TABLE IF NOT EXISTS request_history(history_id TEXT PRIMARY KEY, request_id TEXT, endpoint TEXT,
 requested_at TEXT, completed_at TEXT, status INTEGER, response_sha256 TEXT, raw_path TEXT, error TEXT, attempts INTEGER);
CREATE TABLE IF NOT EXISTS search_log(query_id TEXT PRIMARY KEY, database_name TEXT, query_text TEXT, protocol_version TEXT,
 track TEXT, target_id TEXT, requested_at TEXT, completed_at TEXT, result_count INTEGER, retrieved_count INTEGER,
 response_sha256 TEXT, status TEXT, error TEXT, stopping_reason TEXT, request_id TEXT REFERENCES requests);
CREATE TABLE IF NOT EXISTS study_families(study_family_id TEXT PRIMARY KEY, review_status TEXT NOT NULL, rationale TEXT);
CREATE TABLE IF NOT EXISTS papers(paper_id TEXT PRIMARY KEY, pmid TEXT, pmcid TEXT, doi TEXT, study_family_id TEXT REFERENCES study_families,
 title TEXT, year INTEGER, journal TEXT, publication_type TEXT, access_class TEXT, license TEXT,
 is_retracted INTEGER, retrieved_at TEXT, raw_document_sha256 TEXT, discovery_route TEXT, query_id TEXT,
 full_text_request_id TEXT REFERENCES requests, metadata_json TEXT, training_text_allowed INTEGER DEFAULT 0,
 retraction_status TEXT DEFAULT 'provider_metadata_only');
CREATE TABLE IF NOT EXISTS discoveries(paper_id TEXT REFERENCES papers, query_id TEXT REFERENCES search_log, PRIMARY KEY(paper_id,query_id));
CREATE TABLE IF NOT EXISTS document_sections(section_id TEXT PRIMARY KEY, paper_id TEXT REFERENCES papers, section_name TEXT,
 source_locator TEXT, text TEXT, source_sha256 TEXT);
CREATE TABLE IF NOT EXISTS candidates(candidate_id TEXT PRIMARY KEY, paper_id TEXT REFERENCES papers, pair_id TEXT REFERENCES pairs,
 section_id TEXT REFERENCES document_sections, matched_name_a TEXT, matched_name_b TEXT, dimensions_json TEXT,
 status TEXT DEFAULT 'unreviewed', exclusion_reason TEXT);
CREATE TABLE IF NOT EXISTS screening_votes(candidate_id TEXT REFERENCES candidates, reviewer_id TEXT, stage TEXT,
 decision TEXT CHECK(decision IN ('include','exclude','uncertain','background_only')), reason_code TEXT, created_at TEXT,
 PRIMARY KEY(candidate_id,reviewer_id,stage));
CREATE TABLE IF NOT EXISTS paper_screening_votes(paper_id TEXT REFERENCES papers, reviewer_id TEXT, stage TEXT,
 decision TEXT CHECK(decision IN ('include','exclude','uncertain','background_only')), reason_code TEXT, created_at TEXT,
 PRIMARY KEY(paper_id,reviewer_id,stage));
CREATE TABLE IF NOT EXISTS disease_pair_observations(observation_id TEXT PRIMARY KEY, paper_id TEXT REFERENCES papers,
 study_family_id TEXT REFERENCES study_families, orpha_id_a TEXT REFERENCES diseases, orpha_id_b TEXT REFERENCES diseases,
 entity_type_a TEXT, entity_type_b TEXT, population TEXT, model_system TEXT, study_design TEXT,
 sample_size_a INTEGER, sample_size_b INTEGER, dimension TEXT, ordinal_score INTEGER CHECK(ordinal_score BETWEEN 0 AND 4),
 dimension_applicability INTEGER CHECK(dimension_applicability BETWEEN 0 AND 2), evidence_direction TEXT,
 quality_tier TEXT CHECK(quality_tier IN ('high','moderate','low','very_low')), quality_components_json TEXT,
 extraction_confidence INTEGER CHECK(extraction_confidence BETWEEN 1 AND 3),
 source_section TEXT, source_locator TEXT, source_passage_or_paraphrase TEXT, reviewer_id TEXT,
 rubric_version TEXT, created_at TEXT, review_status TEXT DEFAULT 'unreviewed',
 CHECK(ordinal_score IS NULL OR (length(source_locator)>0 AND source_locator IS NOT NULL
 AND length(source_passage_or_paraphrase)>0 AND source_passage_or_paraphrase IS NOT NULL)),
 CHECK(dimension_applicability != 0 OR ordinal_score IS NULL));
CREATE TABLE IF NOT EXISTS adjudications(observation_group_id TEXT PRIMARY KEY, reviewer_1_score INTEGER, reviewer_2_score INTEGER,
 disagreement_type TEXT, adjudicated_score INTEGER, adjudicator_id TEXT, adjudication_reason TEXT, adjudicated_at TEXT);
CREATE TABLE IF NOT EXISTS pair_consensus(pair_id TEXT REFERENCES pairs, dimension TEXT, consensus_ordinal_score REAL,
 normalized_score REAL, score_interval_low REAL, score_interval_high REAL, independent_study_families INTEGER,
 full_text_studies INTEGER, quality_tier TEXT, heterogeneity_flag INTEGER, control_type TEXT, protocol_version TEXT,
 PRIMARY KEY(pair_id,dimension));
CREATE TABLE IF NOT EXISTS split_assignments(regime TEXT, item_type TEXT, item_id TEXT, split TEXT, component_id TEXT,
 PRIMARY KEY(regime,item_type,item_id));
CREATE TABLE IF NOT EXISTS protocol_amendments(amendment_id TEXT PRIMARY KEY, created_at TEXT, reason TEXT, change_json TEXT);
"""


def connect(out):
    db = sqlite3.connect(out / 'literature.sqlite3')
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    return db


def insert(db, table, row, replace=False):
    verb = 'INSERT OR REPLACE' if replace else 'INSERT OR IGNORE'
    db.execute(f"{verb} INTO {table} ({','.join(row)}) VALUES ({','.join('?' for _ in row)})", tuple(row.values()))


class Client:
    def __init__(self, out, db):
        self.out, self.db, self.last = out, db, 0

    def get(self, endpoint, params=None):
        url = endpoint + ('?' + urllib.parse.urlencode(sorted(params.items())) if params else '')
        rid = digest(url)
        old = self.db.execute('SELECT * FROM requests WHERE request_id=?', (rid,)).fetchone()
        if old and old['status'] == 200:
            data = (self.out / old['raw_path']).read_bytes()
            if digest(data) != old['response_sha256']:
                raise ValueError('Cached response checksum mismatch: ' + rid)
            return data, dict(old)
        if old:
            # A successful retry must not destroy the prior failure's body or timestamps.
            previous = dict(old)
            history_id = digest(rid + previous['requested_at'])
            old_path = self.out / previous['raw_path']
            history_path = Path('raw') / (history_id + '.history.response')
            if old_path.exists():
                (self.out / history_path).write_bytes(old_path.read_bytes())
                previous['raw_path'] = history_path.as_posix()
            insert(self.db, 'request_history', dict(history_id=history_id, **previous))
            self.db.commit()
        started = now()
        error, data, status = None, b'', None
        for attempt in range(1, 5):
            time.sleep(max(0, 0.4 - (time.monotonic() - self.last)))
            self.last = time.monotonic()
            request = urllib.request.Request(url, headers={'User-Agent': 'RareDiseaseEvidence/1.0 (academic corpus pilot)',
                                                           'Accept': 'application/json, application/xml;q=0.9'})
            try:
                with urllib.request.urlopen(request, timeout=45) as response:
                    data, status = response.read(), response.status
                error = None
                break
            except urllib.error.HTTPError as exc:
                status, data, error = exc.code, exc.read(), str(exc)
                if status not in (429, 500, 502, 503, 504):
                    break
                retry = exc.headers.get('Retry-After', '')
                pause = float(retry) if retry.isdigit() else 2 ** attempt + random.random()
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                error = repr(exc)
                pause = 2 ** attempt + random.random()
            if attempt < 4:
                time.sleep(pause)
        raw_path = Path('raw') / (rid + '.response')
        (self.out / raw_path).parent.mkdir(parents=True, exist_ok=True)
        (self.out / raw_path).write_bytes(data)  # Save before parsing, including error bodies.
        record = dict(request_id=rid, endpoint=url, requested_at=started, completed_at=now(), status=status,
                      response_sha256=digest(data), raw_path=raw_path.as_posix(), error=error, attempts=attempt)
        insert(self.db, 'requests', record, replace=True)
        self.db.commit()
        return (data if status == 200 else None), record


def expression(disease):
    return '(' + ' OR '.join('"' + x.replace('"', '') + '"' for x in disease['search_terms']) + ')'


def queries(protocol):
    ds = {d['orpha_id']: d for d in protocol['diseases']}
    date_filter = 'FIRST_PDATE:[' + ' TO '.join(protocol['date_range']) + '] AND SRC:MED'
    for pair in protocol['pairs']:
        pid = '|'.join(pair)
        base = expression(ds[pair[0]]) + ' AND ' + expression(ds[pair[1]])
        yield 'A', pid, base + ' AND ' + date_filter
        for terms in protocol['dimension_terms'].values():
            yield 'A_dimension', pid, base + ' AND (' + terms + ') AND ' + date_filter
        yield 'C', pid, base + ' AND (' + protocol['control_terms'] + ') AND ' + date_filter
    for disease in protocol['diseases']:
        yield 'B', disease['orpha_id'], expression(disease) + ' AND (' + ' OR '.join(protocol['dimension_terms'].values()) + ') AND ' + date_filter


def normalize_doi(doi):
    return re.sub(r'^https?://(?:dx\.)?doi\.org/', '', doi.strip().casefold()).removeprefix('doi:').strip()


def add_paper(db, record, query_id, request):
    pmid = str(record.get('pmid') or (record.get('id') if record.get('source') == 'MED' else '') or '')
    pmcid, doi = record.get('pmcid') or '', normalize_doi(record.get('doi') or '')
    paper_id = 'PMID:' + pmid if pmid else 'PMCID:' + pmcid if pmcid else 'DOI:' + doi
    if paper_id == 'DOI:':
        paper_id = 'TITLE:' + digest(normalize(record.get('title', '')) + str(record.get('pubYear')) + record.get('authorString', '').split(',')[0])
    for field, value in [('pmid', pmid), ('pmcid', pmcid), ('doi', doi)]:
        if value:
            found = db.execute(f'SELECT paper_id FROM papers WHERE {field}=?', (value,)).fetchone()
            if found:
                paper_id = found[0]
                break
    pubs = record.get('pubTypeList', {}).get('pubType', [])
    retracted = any('retracted publication' in x.casefold() for x in pubs) or record.get('isRetracted') == 'Y'
    family = 'UNRESOLVED:' + paper_id
    insert(db, 'study_families', dict(study_family_id=family, review_status='unreviewed',
                                     rationale='Placeholder only; independence has not been established.'))
    journal = record.get('journalInfo', {}).get('journal', {}).get('title', record.get('journalTitle'))
    query = db.execute('SELECT track FROM search_log WHERE query_id=?', (query_id,)).fetchone()
    insert(db, 'papers', dict(paper_id=paper_id, pmid=pmid, pmcid=pmcid, doi=doi, study_family_id=family,
                             title=record.get('title'), year=int(record.get('pubYear') or 0), journal=journal,
                             publication_type=json.dumps(pubs), access_class='retracted_or_removed' if retracted else
                             'abstract_only' if record.get('abstractText') else 'metadata_only',
                             license=record.get('license'), is_retracted=int(retracted), retrieved_at=request['completed_at'],
                             raw_document_sha256=request['response_sha256'], discovery_route=query[0], query_id=query_id,
                             metadata_json=json.dumps(record, ensure_ascii=False)))
    insert(db, 'discoveries', dict(paper_id=paper_id, query_id=query_id))
    return paper_id


def search(client, protocol, track, target, query):
    qid = digest(track + target + query)[:24]
    previous = client.db.execute('SELECT status FROM search_log WHERE query_id=?', (qid,)).fetchone()
    if previous and previous[0] == 'complete':
        return
    data, req = client.get(API + '/search', {'query': query, 'format': 'json', 'resultType': 'core',
                                           'pageSize': protocol['limits']['results_per_query']})
    records, total, error = [], None, req['error']
    if data:
        try:
            parsed = json.loads(data)
            if 'hitCount' not in parsed:
                raise ValueError('Missing hitCount: ' + str(parsed)[:200])
            records, total = parsed.get('resultList', {}).get('result', []), parsed['hitCount']
        except (ValueError, KeyError) as exc:
            error = str(exc)
    insert(client.db, 'search_log', dict(query_id=qid, database_name='Europe PMC', query_text=query,
        protocol_version=VERSION, track=track, target_id=target, requested_at=req['requested_at'],
        completed_at=req['completed_at'], result_count=total, retrieved_count=len(records),
        response_sha256=req['response_sha256'], status='failed' if error else 'complete', error=error,
        stopping_reason='request_failed' if error else 'pilot_page_cap' if total > len(records) else 'query_exhausted',
        request_id=req['request_id']), replace=True)
    for record in records:
        add_paper(client.db, record, qid, req)
    client.db.commit()


def citation_expansion(client, protocol):
    # Provisional OA review seeds, not claims of human inclusion or quality.
    seeds = [dict(r) for r in client.db.execute("""SELECT DISTINCT p.* FROM papers p
      JOIN discoveries d USING(paper_id) JOIN search_log s USING(query_id)
      WHERE p.pmcid!='' AND p.is_retracted=0 AND s.track='A' AND p.publication_type LIKE '%Review%'
      ORDER BY p.paper_id LIMIT ?""", (protocol['limits']['citation_seeds'],))]
    seen = {p['pmid'] for p in seeds}
    for round_no in range(1, protocol['limits']['citation_rounds'] + 1):
        ids = set()
        for seed in seeds:
            for direction in ('references', 'citations'):
                endpoint = API + '/MED/' + seed['pmid'] + '/' + direction
                data, req = client.get(endpoint, {'format': 'json', 'pageSize': protocol['limits']['citation_page_size'], 'page': 1})
                qid = digest(endpoint)[:24]
                parsed = json.loads(data) if data else {}
                singular = 'reference' if direction == 'references' else 'citation'
                records = parsed.get(singular + 'List', {}).get(singular, [])
                for record in records:
                    if record.get('source') == 'MED' and record.get('id'):
                        ids.add(str(record['id']))
                total = parsed.get('hitCount', 0)
                insert(client.db, 'search_log', dict(query_id=qid, database_name='Europe PMC', query_text=endpoint,
                    protocol_version=VERSION, track='citation_' + direction, target_id=seed['paper_id'],
                    requested_at=req['requested_at'], completed_at=req['completed_at'], result_count=total,
                    retrieved_count=len(records), response_sha256=req['response_sha256'],
                    status='complete' if data else 'failed', error=req['error'],
                    stopping_reason='pilot_page_cap' if total > len(records) else 'endpoint_exhausted', request_id=req['request_id']))
                client.db.commit()
        ids -= seen
        # Resolve referenced/citing IDs through the same disease/date eligibility search, in fixed batches.
        disease_filter = '(' + ' OR '.join(expression(d) for d in protocol['diseases']) + ')'
        ids = sorted(ids)
        for offset in range(0, len(ids), 50):
            query = '(' + ' OR '.join('EXT_ID:' + x for x in ids[offset:offset+50]) + ') AND ' + disease_filter
            query += ' AND SRC:MED AND FIRST_PDATE:[' + ' TO '.join(protocol['date_range']) + ']'
            search(client, protocol, 'citation_resolve', 'round:' + str(round_no), query)
        seen.update(ids)
        seeds = [dict(r) for r in client.db.execute("""SELECT DISTINCT p.* FROM papers p JOIN discoveries d USING(paper_id)
          JOIN search_log s USING(query_id) WHERE s.track='citation_resolve' AND p.pmcid!='' AND p.is_retracted=0
          ORDER BY p.paper_id""") if r['pmid'] in ids][:protocol['limits']['citation_seeds']]
        print(f'Citation round {round_no}: {len(ids)} linked PMID candidates', flush=True)


def license_from_xml(root):
    elements = root.findall('./front/article-meta/permissions/license')
    values = []
    for element in elements:
        values.extend(element.attrib.values())
        values.append(plain(element))
        for child in element.iter():
            values.extend(child.attrib.values())
    text = ' '.join(values)
    urls = re.findall(r'https?://creativecommons\.org/(?:licenses/[a-z-]+/\d(?:\.\d)?|publicdomain/zero/\d(?:\.\d)?)/?', text, flags=re.I)
    by_license = any(re.match(r'https?://creativecommons\.org/licenses/by/\d', u, re.I) for u in urls)
    cc0 = any('/publicdomain/zero/' in u.casefold() for u in urls)
    # A CC0 waiver for associated data is not a license for the article body.
    allowed = by_license or (cc0 and not re.search(r'\bdata\b', text, re.I))
    # Reject conflicting NC/ND/SA signals conservatively even if a BY link also appears.
    if re.search(r'by-(?:nc|nd|sa)|noncommercial|non-commercial|no derivatives|sharealike', text, re.I):
        allowed = False
    return ('; '.join(sorted(set(urls))) or text[:1000] or 'unspecified'), allowed


def acquire_text(client, protocol):
    papers = client.db.execute("""SELECT p.*, COUNT(DISTINCT CASE WHEN s.track IN ('A','C','A_dimension') THEN s.target_id END) AS pair_hits
      FROM papers p JOIN discoveries d USING(paper_id) JOIN search_log s USING(query_id)
      WHERE p.pmcid!='' AND p.is_retracted=0 GROUP BY p.paper_id
      ORDER BY pair_hits DESC, p.paper_id""").fetchall()
    attempts = 0
    attempted_ids = {r[0] for r in client.db.execute('SELECT paper_id FROM papers WHERE full_text_request_id IS NOT NULL')}
    for paper in papers:
        meta = json.loads(paper['metadata_json'])
        if meta.get('isOpenAccess') != 'Y':
            continue
        if paper['paper_id'] not in attempted_ids and len(attempted_ids) >= protocol['limits']['full_text_attempts']:
            continue
        attempted_ids.add(paper['paper_id'])
        attempts += 1
        data, req = client.get(API + '/' + paper['pmcid'] + '/fullTextXML')
        client.db.execute('UPDATE papers SET full_text_request_id=? WHERE paper_id=?', (req['request_id'], paper['paper_id']))
        if data:
            try:
                root = ET.fromstring(data)
                license_text, allowed = license_from_xml(root)
                retracted = root.attrib.get('article-type') in ('retracted-article', 'retraction')
                client.db.execute('''UPDATE papers SET access_class=?, license=?, raw_document_sha256=?,
                 training_text_allowed=?, is_retracted=MAX(is_retracted,?) WHERE paper_id=?''',
                 ('retracted_or_removed' if retracted else 'open_full_text_xml', license_text,
                  req['response_sha256'], int(allowed and not retracted), int(retracted), paper['paper_id']))
                if allowed and not retracted:
                    body = root.find('body')
                    if body is not None:
                        parents = {child: parent for parent in body.iter() for child in parent}
                        for i, p in enumerate(body.iter('p'), 1):
                            text = plain(p)
                            if not text:
                                continue
                            ancestor, section = parents.get(p), 'Body'
                            while ancestor is not None:
                                if ancestor.tag == 'sec':
                                    title = ancestor.find('title')
                                    if title is not None:
                                        section = plain(title)
                                        break
                                ancestor = parents.get(ancestor)
                            locator = '/article/body/descendant::p[' + str(i) + ']'
                            insert(client.db, 'document_sections', dict(section_id=digest(paper['paper_id'] + locator)[:28],
                                paper_id=paper['paper_id'], section_name=section, source_locator=locator, text=text,
                                source_sha256=req['response_sha256']))
            except ET.ParseError as exc:
                insert(client.db, 'protocol_amendments', dict(amendment_id='parse:' + req['request_id'], created_at=now(),
                    reason='XML parsing failure; document excluded', change_json=json.dumps({'error': str(exc)})))
        client.db.commit()
        if attempts % 20 == 0:
            print(f'Full text: {attempts}/{protocol["limits"]["full_text_attempts"]} attempts', flush=True)


def mention(text, terms):
    for term in sorted(terms, key=len, reverse=True):
        if re.search(r'(?<!\w)' + re.escape(term) + r'(?!\w)', text, re.I):
            return term
    return None


def make_candidates(db, protocol):
    ds = {d['orpha_id']: d for d in protocol['diseases']}
    for row in db.execute('''SELECT d.* FROM document_sections d JOIN papers p USING(paper_id)
                          WHERE p.training_text_allowed=1 AND p.is_retracted=0''').fetchall():
        hits = {oid: mention(row['text'], d['search_terms']) for oid, d in ds.items()}
        hits = {oid: name for oid, name in hits.items() if name}
        ids = sorted(hits)
        for i, a in enumerate(ids):
            for b in ids[i+1:]:
                if ds[a]['entity_type'] != ds[b]['entity_type']:
                    continue
                pid = a + '|' + b
                insert(db, 'pairs', dict(pair_id=pid, orpha_id_a=a, orpha_id_b=b))
                # These are search hints only. All seven dimensions remain available to reviewers.
                hints = [dim for dim, words in DIMENSIONS.items()
                         if any(w.rstrip('*').strip('"').casefold() in row['text'].casefold() for w in words.split(' OR '))]
                insert(db, 'candidates', dict(candidate_id=digest(row['section_id'] + pid)[:28], paper_id=row['paper_id'],
                    pair_id=pid, section_id=row['section_id'], matched_name_a=hits[a], matched_name_b=hits[b],
                    dimensions_json=json.dumps(hints)))
    db.commit()


def connected_components(edges, nodes=()):
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    for node in nodes:
        find(node)
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return {x: find(x) for x in parent}


def hash_split(key):
    value = int(digest(key)[:8], 16) % 100
    return 'train' if value < 80 else 'validation' if value < 90 else 'test'


def assign_splits(db, protocol):
    db.execute('DELETE FROM split_assignments')
    papers = {p['paper_id']: dict(p) for p in db.execute('SELECT * FROM papers')}
    edges = [('paper:' + p, 'family:' + d['study_family_id']) for p, d in papers.items()]
    paper_pairs = collections.defaultdict(set)
    for row in db.execute('SELECT DISTINCT paper_id,pair_id FROM candidates'):
        paper_pairs[row['paper_id']].add(row['pair_id'])
    # A retrieved paper is also connected to every direct query pair, even if mention matching missed it.
    for row in db.execute("""SELECT d.paper_id,s.target_id FROM discoveries d JOIN search_log s USING(query_id)
                            WHERE s.track IN ('A','A_dimension','C')"""):
        paper_pairs[row[0]].add(row[1])
    for paper, pairs in paper_pairs.items():
        edges.extend(('paper:' + paper, 'pair:' + pair) for pair in pairs)
    # Exact duplicated paragraphs across papers are blocked too, protecting reusable text exports.
    duplicates = collections.defaultdict(list)
    for row in db.execute('SELECT paper_id,text FROM document_sections WHERE length(text)>=200'):
        duplicates[digest(normalize(row[1]))].append(row[0])
    for items in duplicates.values():
        edges.extend(('paper:' + items[0], 'paper:' + x) for x in items[1:])
    components = connected_components(edges)
    for node, component in components.items():
        kind, item = node.split(':', 1)
        insert(db, 'split_assignments', dict(regime='paper_pair_family', item_type=kind, item_id=item,
                                             split=hash_split(component), component_id=component))
    held = {d['orpha_id'] for d in protocol['diseases'] if int(digest('held:' + d['orpha_id'])[:8], 16) % 5 == 0}
    pair_split = {}
    for pair in db.execute('SELECT * FROM pairs'):
        count = sum(oid in held for oid in (pair['orpha_id_a'], pair['orpha_id_b']))
        pair_split[pair['pair_id']] = 'test' if count == 2 else 'train' if count == 0 else 'quarantine'
    ds = {d['orpha_id']: d for d in protocol['diseases']}
    mentions = collections.defaultdict(set)
    for row in db.execute('SELECT paper_id,text FROM document_sections'):
        for oid, disease in ds.items():
            if mention(row['text'], disease['search_terms']):
                mentions[row['paper_id']].add(oid)
    for paper, metadata in papers.items():
        record = json.loads(metadata['metadata_json'] or '{}')
        text = (record.get('title') or '') + ' ' + (record.get('abstractText') or '')
        for oid, disease in ds.items():
            if mention(text, disease['search_terms']):
                mentions[paper].add(oid)
    for paper in papers:
        splits = {pair_split[p] for p in paper_pairs[paper]}
        split = next(iter(splits)) if len(splits) == 1 else 'quarantine'
        if (split == 'train' and mentions[paper] & held) or (split == 'test' and mentions[paper] - held):
            split = 'quarantine'
        insert(db, 'split_assignments', dict(regime='disease_held_out', item_type='paper', item_id=paper,
                                             split=split, component_id=None))
    for pair, split in pair_split.items():
        insert(db, 'split_assignments', dict(regime='disease_held_out', item_type='pair', item_id=pair,
                                             split=split, component_id=None))
    family_times = collections.defaultdict(set)
    for paper in papers.values():
        y = paper['year']
        family_times[paper['study_family_id']].add('train' if 2000 <= y <= 2022 else 'validation' if y == 2023 else 'test' if y >= 2024 else 'quarantine')
    for paper in papers.values():
        values = family_times[paper['study_family_id']]
        insert(db, 'split_assignments', dict(regime='time', item_type='paper', item_id=paper['paper_id'],
            split=next(iter(values)) if len(values) == 1 else 'quarantine', component_id=paper['study_family_id']))
    db.commit()


def validate(db, out):
    errors = []
    for row in db.execute('SELECT * FROM requests'):
        path = out / row['raw_path']
        if not path.exists() or digest(path.read_bytes()) != row['response_sha256']:
            errors.append('raw_checksum:' + row['request_id'])
    for row in db.execute('SELECT * FROM request_history'):
        path = out / row['raw_path']
        if not path.exists() or digest(path.read_bytes()) != row['response_sha256']:
            errors.append('historical_raw_checksum:' + row['history_id'])
    if db.execute('PRAGMA foreign_key_check').fetchall():
        errors.append('foreign_key_check')
    if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
        errors.append('sqlite_integrity')
    for row in db.execute("""SELECT d.paper_id,s.target_id FROM discoveries d JOIN search_log s USING(query_id)
      WHERE s.track IN ('A','A_dimension','C') UNION SELECT paper_id,pair_id FROM candidates"""):
        p = db.execute("SELECT split FROM split_assignments WHERE regime='paper_pair_family' AND item_type='paper' AND item_id=?", (row[0],)).fetchone()
        q = db.execute("SELECT split FROM split_assignments WHERE regime='paper_pair_family' AND item_type='pair' AND item_id=?", (row[1],)).fetchone()
        if not p or not q or p[0] != q[0]:
            errors.append('split_leakage:' + row[0])
    if db.execute("SELECT COUNT(*) FROM papers WHERE training_text_allowed=1 AND (is_retracted=1 OR access_class!='open_full_text_xml')").fetchone()[0]:
        errors.append('ineligible_training_text')
    for row in db.execute("""SELECT p.study_family_id,COUNT(DISTINCT s.split) FROM papers p
        JOIN split_assignments s ON s.item_id=p.paper_id AND s.item_type='paper'
        WHERE s.regime='paper_pair_family' GROUP BY p.study_family_id HAVING COUNT(DISTINCT s.split)>1"""):
        errors.append('family_split_leakage:' + row[0])
    for row in db.execute('SELECT * FROM disease_pair_observations WHERE ordinal_score IS NOT NULL'):
        if row['dimension'] not in DIMENSIONS or row['dimension_applicability'] not in (1, 2):
            errors.append('invalid_dimension:' + row['observation_id'])
        if not row['source_locator'] or not row['source_passage_or_paraphrase']:
            errors.append('missing_locator:' + row['observation_id'])
    retracted = db.execute('''SELECT o.observation_id FROM disease_pair_observations o JOIN papers p USING(paper_id)
                             WHERE p.is_retracted=1 AND o.review_status='approved' ''').fetchall()
    errors.extend('retracted_observation:' + r[0] for r in retracted)
    return errors


def export(out, db, protocol):
    exports = out / 'exports'
    exports.mkdir(exist_ok=True)
    counts = collections.Counter()
    with (exports / 'text_training.jsonl').open('w', encoding='utf-8') as handle:
        for row in db.execute("""SELECT d.*,p.pmid,p.pmcid,p.doi,p.license,p.study_family_id,p.title,p.year,p.journal,
           p.metadata_json,s.split FROM document_sections d
           JOIN papers p USING(paper_id) JOIN split_assignments s ON s.item_id=p.paper_id
           AND s.item_type='paper' AND s.regime='paper_pair_family'
           WHERE p.training_text_allowed=1 AND p.is_retracted=0 ORDER BY p.paper_id,d.source_locator"""):
            record = dict(row)
            record['authors'] = json.loads(record.pop('metadata_json')).get('authorString')
            if len(record['text']) < 80:
                continue
            record['purpose'] = 'domain_adaptation_or_retrieval; not disease_similarity_ground_truth'
            record['split_status'] = 'provisional_pending_study_family_review'
            record['source_url'] = 'https://europepmc.org/articles/' + record['pmcid']
            record['text_transformations'] = 'XML markup removed; whitespace normalized; body paragraphs only'
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            counts[record['split']] += 1
    # Only approved labels belong here. This pipeline deliberately cannot manufacture review votes.
    (exports / 'supervised_training.jsonl').write_text('', encoding='utf-8')
    queue_sql = """SELECT c.*,d.section_name,d.source_locator,d.text AS source_passage,p.title,p.pmid,p.pmcid,
       p.license,p.study_family_id,p.year,p.access_class FROM candidates c JOIN document_sections d USING(section_id)
       JOIN papers p ON p.paper_id=c.paper_id WHERE p.training_text_allowed=1 AND p.is_retracted=0
       ORDER BY c.pair_id,c.paper_id,d.source_locator"""
    with (exports / 'review_queue.jsonl').open('w', encoding='utf-8') as handle:
        for row in db.execute(queue_sql):
            record = dict(row)
            record.update(dict(ordinal_score=None, quality_tier=None, population=None, model_system=None,
                               sample_size_a=None, sample_size_b=None, reviewer_id=None,
                               dimensions_to_review=list(DIMENSIONS), rubric_version=protocol['rubric_version']))
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    with (exports / 'screening_queue.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['paper_id', 'title', 'year', 'pmid', 'pmcid', 'access_class', 'reviewer_id', 'decision', 'reason_code'])
        for row in db.execute('SELECT paper_id,title,year,pmid,pmcid,access_class FROM papers ORDER BY paper_id'):
            writer.writerow(list(row) + ['', '', ''])
    with (exports / 'pair_search_coverage.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['pair_id', 'queries_planned', 'queries_complete', 'capped_queries', 'distinct_discovered_papers',
                         'candidate_full_text_passages', 'stopping_status'])
        for pair in protocol['pairs']:
            pid = '|'.join(pair)
            complete = db.execute("SELECT COUNT(*) FROM search_log WHERE target_id=? AND status='complete'", (pid,)).fetchone()[0]
            capped = db.execute("SELECT COUNT(*) FROM search_log WHERE target_id=? AND stopping_reason='pilot_page_cap'", (pid,)).fetchone()[0]
            discovered = db.execute('SELECT COUNT(DISTINCT paper_id) FROM discoveries JOIN search_log USING(query_id) WHERE target_id=?', (pid,)).fetchone()[0]
            passages = db.execute('SELECT COUNT(*) FROM candidates WHERE pair_id=?', (pid,)).fetchone()[0]
            writer.writerow([pid, 9, complete, capped, discovered, passages, 'pilot_only; independent_screening_and_saturation_not_complete'])
    errors = validate(db, out)
    def histogram(sql):
        return {str(r[0]): r[1] for r in db.execute(sql)}
    manifest = {
        'protocol_version': VERSION, 'generated_at': now(), 'protocol_sha256': digest((out / 'protocol.json').read_bytes()),
        'release_type': 'calibration_acquisition_corpus',
        'supervised_release_approved': False, 'primary_benchmark_approved': False,
        'row_counts': {t: db.execute('SELECT COUNT(*) FROM ' + t).fetchone()[0] for t in
                       ['diseases','pairs','papers','requests','request_history','search_log','document_sections','candidates',
                        'screening_votes','paper_screening_votes','disease_pair_observations','adjudications','pair_consensus']},
        'access_classes': histogram('SELECT access_class,COUNT(*) FROM papers GROUP BY access_class'),
        'text_training_papers': db.execute('''SELECT COUNT(DISTINCT d.paper_id) FROM document_sections d
                                             JOIN papers p USING(paper_id) WHERE p.training_text_allowed=1 AND p.is_retracted=0''').fetchone()[0],
        'text_training_paragraphs_by_split': dict(counts),
        'query_status': histogram('SELECT status,COUNT(*) FROM search_log GROUP BY status'),
        'historical_request_failures': sum(db.execute('SELECT COUNT(*) FROM ' + t + ' WHERE status IS NULL OR status!=200').fetchone()[0]
                                           for t in ['requests', 'request_history']),
        'query_stopping_reasons': histogram('SELECT stopping_reason,COUNT(*) FROM search_log GROUP BY stopping_reason'),
        'control_distribution': histogram('SELECT control_type,COUNT(*) FROM pairs GROUP BY control_type'),
        'dimension_coverage': {d: 0 for d in DIMENSIONS}, 'score_distribution': {},
        'reviewer_agreement': {'status': 'not_measured', 'reviewers': 0, 'kappa': None},
        'mapping_coverage': {'dictionary_entities': len(protocol['diseases']), 'paper_mappings_adjudicated': 0},
        'split_integrity': {'mechanical_checks_passed': not errors, 'errors': errors,
                            'family_independence_verified': False, 'all_splits_provisional': True},
        'rejections': protocol['seed_selection_rejections'],
        'blocked_supervised_release_reasons': ['No independent screening or calibrated extraction.',
            'No reviewed study families, adjudicated mappings, controls, or labels.',
            'Search and full-text budgets are capped; systematic stopping criteria not met.',
            'Minimum held-out evidence and control balance not met.'],
        'file_checksums': {p.name: digest(p.read_bytes()) for p in exports.iterdir() if p.is_file()},
        'collector_sha256': digest(Path(__file__).read_bytes()),
    }
    write_json(out / 'release_manifest.json', manifest)
    if errors:
        raise ValueError('Acquisition integrity failed: ' + repr(errors[:10]))
    print(json.dumps(manifest, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['register', 'run', 'export', 'validate'])
    parser.add_argument('--output', type=Path, default=DEFAULT)
    args = parser.parse_args()
    out = args.output.resolve()
    if args.command == 'register':
        register(out)
        return
    protocol_bytes = (out / 'protocol.json').read_bytes()
    if digest(protocol_bytes) != (out / 'protocol.sha256').read_text().strip():
        raise ValueError('Registered protocol changed; create a versioned amendment instead.')
    protocol = json.loads(protocol_bytes)
    if digest((out / 'rubric.md').read_bytes()) != protocol['rubric_sha256']:
        raise ValueError('Registered rubric changed.')
    db = connect(out)
    if args.command == 'validate':
        errors = validate(db, out)
        print(json.dumps({'errors': errors, 'passed': not errors}))
        raise SystemExit(bool(errors))
    if args.command == 'run':
        for disease in protocol['diseases']:
            insert(db, 'diseases', dict(orpha_id=disease['orpha_id'], preferred_name=disease['preferred_name'],
                                       entity_type=disease['entity_type'], metadata_json=json.dumps(disease)))
        for pair in protocol['pairs']:
            insert(db, 'pairs', dict(pair_id='|'.join(pair), orpha_id_a=pair[0], orpha_id_b=pair[1]))
        db.commit()
        client = Client(out, db)
        planned = list(queries(protocol))
        for i, (track, target, query) in enumerate(planned, 1):
            search(client, protocol, track, target, query)
            if i % 20 == 0:
                print(f'Searches: {i}/{len(planned)}; papers: {db.execute("SELECT COUNT(*) FROM papers").fetchone()[0]}', flush=True)
        citation_expansion(client, protocol)
        acquire_text(client, protocol)
        make_candidates(db, protocol)
    assign_splits(db, protocol)
    export(out, db, protocol)
    db.close()


if __name__ == '__main__':
    main()
