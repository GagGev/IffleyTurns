"""Fit the project's TF-IDF/SVD/ridge architecture on acquired paper text.

Self-supervised target: the embedding of other paragraphs in the same paper.
Evaluation target: source-paper retrieval, never biological similarity.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import joblib
import numpy as np
import sklearn
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / '.data/literature_acquisition/v1'
OUT = ROOT / 'experiments/artifacts/acquired_corpus_retrieval_v1'
SEED = 42
SKIP_SECTIONS = re.compile(r'acknowledg|conflict|funding|author.? contribution|data availability|ethic|consent|supplement|abbreviation|publisher|declaration', re.I)


def sha(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def clean(text):
    text = re.sub(r'https?://\S+|\b10\.\d{4,9}/\S+|\bPM(?:C)?ID\s*:?\s*\d+', ' ', text, flags=re.I)
    text = re.sub(r'\[[\d\s,;\-–]+\]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def partition(rows):
    """Deterministic, disjoint query/gallery halves within each paper."""
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row['paper_id']].append(row)
    queries, gallery, excluded = [], [], []
    for paper, items in sorted(groups.items()):
        items = sorted(items, key=lambda r: sha('partition-v1:' + r['section_id']))
        if len(items) < 4:
            excluded.append(paper)
            continue
        gallery.extend(items[::2])
        queries.extend(items[1::2])
    assert not ({r['section_id'] for r in queries} & {r['section_id'] for r in gallery})
    return queries, gallery, excluded


def prepare(rows):
    filtered, reasons = [], collections.Counter()
    for row in rows:
        if row['split'] not in ('train', 'test'):
            reasons['other_split'] += 1
            continue
        if SKIP_SECTIONS.search(row['section_name']):
            reasons['administrative_section'] += 1
            continue
        row = dict(row, text=clean(row['text']))
        if len(row['text'].split()) < 40:
            reasons['fewer_than_40_words'] += 1
            continue
        filtered.append(row)
    owners = collections.defaultdict(set)
    for row in filtered:
        owners[sha(row['text'].casefold())].add(row['paper_id'])
    seen, unique = set(), []
    for row in sorted(filtered, key=lambda r: r['section_id']):
        key = sha(row['text'].casefold())
        if len(owners[key]) > 1:
            reasons['text_shared_between_papers'] += 1
        elif key in seen:
            reasons['duplicate_within_paper'] += 1
        else:
            unique.append(row)
            seen.add(key)
    return unique, dict(reasons)


def centroid_matrix(matrix, rows, paper_ids):
    lookup = {p: i for i, p in enumerate(paper_ids)}
    counts = collections.Counter(r['paper_id'] for r in rows)
    pooling = sparse.csr_matrix(([1.0 / counts[r['paper_id']] for r in rows],
                               ([lookup[r['paper_id']] for r in rows], np.arange(len(rows)))),
                              shape=(len(paper_ids), len(rows)))
    return normalize(pooling @ matrix)


def metrics(ranks):
    r = np.asarray(ranks, dtype=float)
    return {'queries': len(r), 'recall_at_1': float(np.mean(r <= 1)),
            'recall_at_5': float(np.mean(r <= 5)), 'recall_at_10': float(np.mean(r <= 10)),
            'mrr': float(np.mean(1 / r)), 'ndcg_at_10': float(np.mean(np.where(r <= 10, 1 / np.log2(r + 1), 0))),
            'median_rank': float(np.median(r))}


def evaluate(scores, queries, papers):
    ranks, per_paper, predictions = [], collections.defaultdict(list), []
    for row, similarities in zip(queries, scores):
        order = np.argsort(-similarities, kind='stable')
        target = papers.index(row['paper_id'])
        rank = int(np.flatnonzero(order == target)[0]) + 1
        ranks.append(rank)
        per_paper[row['paper_id']].append(rank)
        predictions.append({'section_id': row['section_id'], 'source_locator': row['source_locator'],
                            'paper_id': row['paper_id'], 'correct_paper_rank': rank,
                            'top_10_paper_ids': [papers[i] for i in order[:10]],
                            'top_10_cosines': [float(similarities[i]) for i in order[:10]]})
    by_paper = {p: metrics(values) for p, values in per_paper.items()}
    macro = {key: float(np.mean([m[key] for m in by_paper.values()]))
             for key in metrics(ranks) if key not in ('queries', 'median_rank')}
    return {'micro': metrics(ranks), 'macro_over_papers': macro, 'per_paper': by_paper}, predictions


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    input_path = SOURCE / 'exports/text_training.jsonl'
    source_hash = sha(input_path.read_bytes())
    # Record the task/targets/settings before fitting or evaluating. No parameter sweep.
    plan = {'experiment': 'acquired-corpus-source-paper-retrieval-v1', 'registered_at': datetime.now(timezone.utc).isoformat(),
            'source_sha256': source_hash, 'architecture_reference': 'models/svd_embeddings.py',
            'task': 'paragraph-to-source-paper retrieval using self-supervised within-paper targets',
            'not_evaluated': 'seven-dimensional disease similarity: no reviewed targets available',
            'split': 'existing provisional paper_pair_family split; no resplitting',
            'within_paper': 'SHA256 deterministic alternating gallery/query paragraphs; minimum four usable paragraphs',
            'text_filters': '>=40 words; exclude administrative sections; strip URLs, DOI/PMID and numeric citations; remove exact duplicated text',
            'model': {'tfidf': 'word 1-2 grams; min_df=2; max_df=0.95; max_features=30000; sublinear_tf',
                      'svd_components': 128, 'svd_iterations': 7, 'ridge_alpha': 10.0,
                      'ridge_scaler': 'StandardScaler', 'random_seed': SEED,
                      'weighting': 'each training paper contributes equal total ridge sample weight'},
            'fit_scope': 'TF-IDF and SVD on train paper text only; ridge on train query embeddings -> disjoint train gallery centroids',
            'index_scope': 'train and held-out gallery text may be indexed with frozen transforms; held-out query paragraphs never in index',
            'baselines': ['raw TF-IDF cosine', 'SVD cosine', 'uniform random expected rank'],
            'primary_metric_for_this_exploratory_task': 'paper-macro mean reciprocal rank',
            'secondary_metrics': ['Recall@1','Recall@5','Recall@10','nDCG@10','per-paper ranks'],
            'limitations': ['Only two held-out papers and no validation split.',
                            'No hyperparameter selection or confidence/significance claim.',
                            'Source identity is an automatic retrieval target, not clinical similarity evidence.',
                            'Family review remains incomplete; paper split is provisional.',
                            'This new experiment does not amend or overwrite the source review protocol.']}
    plan_path = OUT / 'experiment_plan.json'
    if plan_path.exists():
        prior = json.loads(plan_path.read_text(encoding='utf-8'))
        if prior['source_sha256'] != source_hash:
            raise ValueError('Source changed. Use a new versioned experiment directory.')
        plan = prior
    else:
        save_json(plan_path, plan)

    raw = [json.loads(line) for line in input_path.read_text(encoding='utf-8').splitlines()]
    db = sqlite3.connect(f'file:{(SOURCE / "literature.sqlite3").as_posix()}?mode=ro', uri=True)
    label_count = db.execute('SELECT COUNT(*) FROM disease_pair_observations WHERE ordinal_score IS NOT NULL').fetchone()[0]
    if label_count:
        raise ValueError('Reviewed-target availability changed; preregister an appropriate supervised experiment.')
    lookup = {row[0]: (row[1], row[2]) for row in db.execute("SELECT item_id,split,component_id FROM split_assignments WHERE regime='paper_pair_family' AND item_type='paper'")}
    for row in raw:
        assert row['split'] == lookup[row['paper_id']][0]
    train_components = {lookup[r['paper_id']][1] for r in raw if r['split'] == 'train'}
    test_components = {lookup[r['paper_id']][1] for r in raw if r['split'] == 'test'}
    assert not train_components & test_components
    db.close()

    rows, removed = prepare(raw)
    queries, gallery, excluded = partition(rows)
    train_q = [r for r in queries if r['split'] == 'train']
    test_q = [r for r in queries if r['split'] == 'test']
    train_rows = [r for r in queries + gallery if r['split'] == 'train']
    papers = sorted({r['paper_id'] for r in gallery})
    paper_lookup = {p: i for i, p in enumerate(papers)}
    train_papers = {r['paper_id'] for r in train_rows}
    test_papers = {r['paper_id'] for r in test_q}
    assert train_papers.isdisjoint(test_papers) and len(test_papers) >= 1
    print(f'Fit: {len(train_rows)} paragraphs, {len(train_papers)} papers; test: {len(test_q)} queries, {len(test_papers)} papers; index: {len(papers)} papers', flush=True)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=.95, max_features=30000,
                                 sublinear_tf=True, strip_accents='unicode', dtype=np.float64)
    train_x = vectorizer.fit_transform([r['text'] for r in train_rows])
    svd = TruncatedSVD(n_components=min(128, min(train_x.shape) - 1), n_iter=7, random_state=SEED)
    svd.fit(train_x)
    gx = vectorizer.transform([r['text'] for r in gallery])
    tqx = vectorizer.transform([r['text'] for r in train_q])
    eqx = vectorizer.transform([r['text'] for r in test_q])
    latent_gallery = normalize(svd.transform(gx))
    dense_index = np.asarray(centroid_matrix(latent_gallery, gallery, papers))
    sparse_index = centroid_matrix(gx, gallery, papers)
    latent_train = normalize(svd.transform(tqx))
    latent_test = normalize(svd.transform(eqx))
    targets = dense_index[[paper_lookup[r['paper_id']] for r in train_q]]
    paper_counts = collections.Counter(r['paper_id'] for r in train_q)
    weights = np.array([1 / paper_counts[r['paper_id']] for r in train_q])
    weights /= weights.mean()
    calibrator = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    calibrator.fit(latent_train, targets, ridge__sample_weight=weights)

    scores = {'tfidf': (eqx @ sparse_index.T).toarray(),
              'svd': latent_test @ dense_index.T,
              'svd_ridge': normalize(calibrator.predict(latent_test)) @ dense_index.T}
    results, predictions = {}, {}
    for name, score in scores.items():
        results[name], predictions[name] = evaluate(score, test_q, papers)
    n = len(papers)
    chance = {'recall_at_1': 1/n, 'recall_at_5': min(5/n, 1), 'recall_at_10': min(10/n, 1),
              'mrr': sum(1 / r for r in range(1, n + 1)) / n,
              'ndcg_at_10': sum(1 / np.log2(r + 1) for r in range(1, min(10, n) + 1)) / n}
    payload = {'created_at': datetime.now(timezone.utc).isoformat(), 'source_sha256': source_hash,
               'code_sha256': sha(Path(__file__).read_bytes()), 'task': plan['task'], 'biological_similarity_metrics': None,
               'versions': {'python': sys.version, 'numpy': np.__version__, 'sklearn': sklearn.__version__},
               'counts': {'original_paragraphs': len(raw), 'training_paragraphs': len(train_rows),
                          'training_queries': len(train_q), 'training_papers': len(train_papers),
                          'test_queries': len(test_q), 'test_papers': len(test_papers), 'gallery_papers': n,
                          'gallery_paragraphs': len(gallery), 'vocabulary_terms': len(vectorizer.vocabulary_),
                          'excluded_papers_too_few_paragraphs': len(excluded)},
               'filter_rejections': removed, 'models': results, 'uniform_random_expectation': chance,
               'explained_training_variance': float(svd.explained_variance_ratio_.sum()),
               'integrity': {'train_test_papers_disjoint': True, 'train_test_components_disjoint': True,
                             'query_gallery_sections_disjoint': True, 'held_out_text_used_for_fitting': False,
                             'clinical_labels_used': 0}, 'limitations': plan['limitations']}
    save_json(OUT / 'metrics.json', payload)
    joblib.dump({'vectorizer': vectorizer, 'svd': svd, 'ridge': calibrator,
                 'source_sha256': source_hash, 'task': plan['task'], 'training_paper_ids': sorted(train_papers)}, OUT / 'model.joblib')
    np.savez_compressed(OUT / 'gallery_index.npz', paper_ids=np.asarray(papers), latent_centroids=dense_index)
    assignments = [{'section_id': r['section_id'], 'paper_id': r['paper_id'], 'split': r['split'], 'role': role}
                   for role, group in [('query', queries), ('gallery', gallery)] for r in group]
    with (OUT / 'paragraph_assignments.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(assignments[0]))
        writer.writeheader(); writer.writerows(assignments)
    with (OUT / 'test_predictions.jsonl').open('w', encoding='utf-8') as handle:
        for model, rows_for_model in predictions.items():
            for row in rows_for_model:
                handle.write(json.dumps(dict(model=model, **row)) + '\n')
    make_report(payload, test_q)
    print(json.dumps(payload, indent=2), flush=True)


def make_report(payload, test_q):
    counts, results = payload['counts'], payload['models']
    lines = ['# Acquired-literature model experiment', '',
             'This is source-paper retrieval, not a test of disease-similarity accuracy.', '',
             'Adapted the existing TF-IDF/SVD/ridge architecture to the licensed corpus. '
             'TF-IDF and 128-dimensional SVD are fitted on training text; ridge predicts the '
             'centroid of other paragraphs from the same training paper. Gallery and query paragraphs '
             'are disjoint. The fixed ridge alpha is 10; there was no hyperparameter search.', '',
             f"Training: {counts['training_paragraphs']:,} paragraphs from {counts['training_papers']} papers. "
             f"Evaluation: {counts['test_queries']} queries from {counts['test_papers']} held-out papers, "
             f"ranked against {counts['gallery_papers']} candidate source papers.", '',
             'Results below average each paper equally (paper-macro); paragraphs are not independent studies.', '']
    for name, result in results.items():
        m = result['macro_over_papers']
        lines.append(f"- **{name}**: MRR {m['mrr']:.3f}; Recall@1 {m['recall_at_1']:.1%}; "
                     f"Recall@5 {m['recall_at_5']:.1%}; Recall@10 {m['recall_at_10']:.1%}; nDCG@10 {m['ndcg_at_10']:.3f}.")
    c = payload['uniform_random_expectation']
    lines.extend(['', f"Uniform-random expected MRR: {c['mrr']:.3f}; Recall@1: {c['recall_at_1']:.1%}.", '',
                  '## Per-paper results', ''])
    for paper in sorted({r['paper_id'] for r in test_q}):
        row = next(r for r in test_q if r['paper_id'] == paper)
        lines.extend([f"### {row['title']}", '', f"[Source]({row['source_url']}); {paper}.", ''])
        for name, result in results.items():
            m = result['per_paper'][paper]
            lines.append(f"- {name}: {m['queries']} queries; MRR {m['mrr']:.3f}; Recall@1 {m['recall_at_1']:.1%}; Recall@5 {m['recall_at_5']:.1%}.")
        lines.append('')
    lines.extend(['## Interpretation and limits', '',
                  'The ridge head is trained on known paper centroids and can overfit source-specific structure. '
                  'Its held-out retrieval scores must be judged against the simpler TF-IDF and SVD baselines, '
                  'not just chance. Strong source retrieval does not demonstrate biological similarity.', '',
                  *['- ' + x for x in payload['limitations']], '',
                  'The held-out papers both concern skeletal conditions, so these results are particularly '
                  'narrow. We do not infer broad generalization, statistical significance, or seven-dimension '
                  'prediction accuracy. Unknown diseases/pairs were never labelled unrelated.', '',
                  '## Reproduction and outputs', '',
                  'Run `python experiments/04_acquired_corpus_retrieval.py` from the repository root.', '',
                  '- `experiment_plan.json`: fixed task, split, filters and model settings recorded before fitting.',
                  '- `model.joblib`: fitted text encoder and ridge head.',
                  '- `gallery_index.npz`: frozen-transform retrieval index, including held-out gallery material.',
                  '- `metrics.json`: aggregate and per-paper results, provenance and package versions.',
                  '- `test_predictions.jsonl`: ranked source papers per held-out query.',
                  '- `paragraph_assignments.csv`: reproducible query/gallery assignments.', '',
                  'The acquisition database, source protocol and original production models were not modified.'])
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
