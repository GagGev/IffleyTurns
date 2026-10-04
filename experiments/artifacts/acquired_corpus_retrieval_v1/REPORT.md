# Acquired-literature model experiment

This is source-paper retrieval, not a test of disease-similarity accuracy.

Adapted the existing TF-IDF/SVD/ridge architecture to the licensed corpus. TF-IDF and 128-dimensional SVD are fitted on training text; ridge predicts the centroid of other paragraphs from the same training paper. Gallery and query paragraphs are disjoint. The fixed ridge alpha is 10; there was no hyperparameter search.

Training: 3,908 paragraphs from 104 papers. Evaluation: 28 queries from 2 held-out papers, ranked against 106 candidate source papers.

Results below average each paper equally (paper-macro); paragraphs are not independent studies.

- **tfidf**: MRR 0.926; Recall@1 89.0%; Recall@5 96.7%; Recall@10 100.0%; nDCG@10 0.944.
- **svd**: MRR 0.861; Recall@1 75.1%; Recall@5 96.7%; Recall@10 96.7%; nDCG@10 0.887.
- **svd_ridge**: MRR 0.902; Recall@1 85.6%; Recall@5 100.0%; Recall@10 100.0%; nDCG@10 0.926.

Uniform-random expected MRR: 0.049; Recall@1: 0.9%.

## Per-paper results

### What the pediatric endocrinologist needs to know about skeletal dysplasia, a primer.

[Source](https://europepmc.org/articles/PMC10477785); PMID:37675393.

- tfidf: 13 queries; MRR 0.910; Recall@1 84.6%; Recall@5 100.0%.
- svd: 13 queries; MRR 0.885; Recall@1 76.9%; Recall@5 100.0%.
- svd_ridge: 13 queries; MRR 0.891; Recall@1 84.6%; Recall@5 100.0%.

### The Diagnostic Odyssey in Children and Adolescents With X-linked Hypophosphatemia: Population-Based, Case-Control Study.

[Source](https://europepmc.org/articles/PMC11244174); PMID:38335127.

- tfidf: 15 queries; MRR 0.942; Recall@1 93.3%; Recall@5 93.3%.
- svd: 15 queries; MRR 0.838; Recall@1 73.3%; Recall@5 93.3%.
- svd_ridge: 15 queries; MRR 0.913; Recall@1 86.7%; Recall@5 100.0%.

## Interpretation and limits

The ridge head is trained on known paper centroids and can overfit source-specific structure. Its held-out retrieval scores must be judged against the simpler TF-IDF and SVD baselines, not just chance. Strong source retrieval does not demonstrate biological similarity.

- Only two held-out papers and no validation split.
- No hyperparameter selection or confidence/significance claim.
- Source identity is an automatic retrieval target, not clinical similarity evidence.
- Family review remains incomplete; paper split is provisional.
- This new experiment does not amend or overwrite the source review protocol.

The held-out papers both concern skeletal conditions, so these results are particularly narrow. We do not infer broad generalization, statistical significance, or seven-dimension prediction accuracy. Unknown diseases/pairs were never labelled unrelated.

## Reproduction and outputs

Run `python experiments/04_acquired_corpus_retrieval.py` from the repository root.

- `experiment_plan.json`: fixed task, split, filters and model settings recorded before fitting.
- `model.joblib`: fitted text encoder and ridge head.
- `gallery_index.npz`: frozen-transform retrieval index, including held-out gallery material.
- `metrics.json`: aggregate and per-paper results, provenance and package versions.
- `test_predictions.jsonl`: ranked source papers per held-out query.
- `paragraph_assignments.csv`: reproducible query/gallery assignments.

The acquisition database, source protocol and original production models were not modified.
