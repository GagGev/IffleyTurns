# Semantic similarity benchmark

`02_semantic_similarity_benchmark.py` is an isolated experiment that tests
whether better feature semantics improve the current literature-score task.
It reads the existing `.data` inputs but refuses to write outside the root
`experiments/` directory.

## Applied changes

- Expands direct HPO annotations through the HPO `is_a` hierarchy.
- Weights expanded HPO profiles by corpus information content.
- Replaces broad Open Targets gene sets with curated genes and Open Targets
  associations scoring at least 0.5.
- Adds ontology-parent, full treatment, disease-type, and disorder-group
  comparisons.
- Uses full sparse TF-IDF cosine over Orphanet descriptions instead of lossy
  low-dimensional SVD vectors.
- Reduces the learned representation from 249 mostly hashed values to 93 or 95
  interpretable values.
- Selects models and the inclusion of description text on validation data only.
- Reports a paired bootstrap interval on the untouched test examples.
- Adds a retrieval stress test using body-system-matched, unobserved controls.

The matched controls are not claimed to be clinically unrelated. They test
whether known high-similarity literature edges rank above plausible candidates;
they do not replace expert negative labels.

## Run

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
python experiments/02_semantic_similarity_benchmark.py
```

Outputs are written to `experiments/artifacts/semantic_similarity_v1/`:

- `metrics.json`: complete model selection, test metrics, uncertainty, and
  retrieval results.
- `feature_manifest.json`: source filters and every feature name.
- `test_predictions.csv`: paired current/improved predictions.
- `matched_control_rankings.csv`: row-level retrieval candidates and scores.
- `models.joblib`: fitted benchmark estimators.
- `report.html`: self-contained visual result.

## Interpretation

The primary comparison retains the current paper-averaged labels and pair split
to isolate the effect of the representation. This makes the result directly
comparable, but it cannot fix subjective labels, abstract-only review, or the
scarcity of verified unrelated pairs. A meaningful RMSE improvement therefore
shows better recovery of the current labels—not clinical validation.

## Dimension-aligned follow-up

`03_dimension_aligned_ablation.py` applies the compact features to the four
dimension targets with usable test sizes. It holds the estimator constant:
both the current and aligned representations use the production weighted ridge
procedure and alpha grid. It also removes test pairs sharing a source paper
with train or validation as a sensitivity check.

The follow-up does **not** support replacing the current dimension features
wholesale:

- Macro RMSE worsened from 0.1801 to 0.1864.
- Macro Spearman fell from 0.272 to 0.225.
- Only mechanism RMSE improved, slightly: 0.1766 to 0.1761.
- Mechanism ranking improved substantially (Spearman 0.078 to 0.436), including
  after paper blocking (0.020 to 0.516), but its R² remains negative.
- Phenotype and genetic performance worsened, indicating that hierarchy
  expansion and gene filtering alone do not resolve their label mismatch.

Artifacts are under `experiments/artifacts/dimension_aligned_v1/`, including a
self-contained `report.html`, full metrics, predictions, and paper-blocked
comparisons.
