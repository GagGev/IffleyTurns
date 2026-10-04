# Rare-disease similarity, v4

v4 deliberately has one architecture and one score:

```text
9 drug-free sparse modalities
  -> learned 64-d projection per modality
  -> availability-aware attention pooling
  -> normalized 128-d disease embedding
  -> cosine similarity
```

There is no pair classifier, linear fusion, ensemble, forecast stacker, or
regulatory-history path. The nine inputs are phenotype, gene, pathway,
ontology, name, clinical text, inheritance, onset, and prevalence. Known drug
words are removed from clinical text.

## Leakage control

The embedding is trained with sibling and shared-causal-gene auxiliary
relations. Sibling batches hide ontology and name; gene batches hide gene and
pathway. A disease uses the same deterministic train/validation/test split as
v2:

- training pairs have two train endpoints;
- validation relations select the epoch;
- every pair containing a test disease is untouched until the final
  diagnostic evaluation.

The global curated-relation benchmark still reports all sampled queries for
compatibility, but the test-query row is the meaningful held-out v4 result.
The v2 production model in that benchmark was fitted on all catalogue labels,
so its row is not a held-out comparison.

## Run

```bash
python v4/train.py
python -m pytest v4/tests global_eval/tests -q
python global_eval/run.py --rescore \
  --models v2 v2_drugfree v4_embedding \
  --only paper_pairs relations \
  --output-dir v4/.data/evaluation/global_eval
```

The selected-model command runs `v2_side.py` and `v4_side.py` only. It does
not import or execute v3.

Artifacts:

- `v4/.data/model/embedding_model.joblib`: fitted encoders, sparse catalogue,
  network state, and embeddings;
- `v4/.data/model/embeddings.parquet`: one 128-d row per disease;
- `v4/.data/evaluation/training_summary.json`: split, relation, coverage, and
  held-out diagnostics;
- `v4/.data/evaluation/global_eval/`: focused v2-v4 report and metrics.
