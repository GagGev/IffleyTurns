# Rare-disease similarity, v4

v4 scores a pair with a leakage-safe hybrid of two drug-free views:

```text
9 drug-free sparse modalities
  -> learned 128-d embedding cosine
  +  v2-style non-negative fusion of the nine cosines
  -> scaled and mixed on validation MAP
```

The embedding path is unchanged. The fusion path is v2's logistic over
per-modality cosines, refit on v4's train/train pairs with the same task
masks. There is no forecast stacker and v4 does not import v2. The nine
inputs are phenotype, gene, pathway, ontology, name, clinical text,
inheritance, onset, and prevalence. Known drug words are removed from
clinical text.

## Leakage control

The embedding and the fusion are both trained with sibling,
shared-causal-gene, and HPO-neighbour auxiliary relations. Sibling batches
hide ontology and name; gene batches hide gene and pathway; HPO-neighbour
batches hide phenotype. A sampled-softmax objective uses uniform and online
hard negatives while excluding positives from every task. A disease uses the
same deterministic train/validation/test split as v2:

- training pairs have two train endpoints;
- full-gallery validation MAP selects the embedding epoch and the fusion mix;
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
