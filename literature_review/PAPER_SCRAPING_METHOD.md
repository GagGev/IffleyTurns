# Paper scraping method for the similarity models

Written for: the team building the training labels for the rare-disease similarity models.

This describes how to collect the literature ground truth so it matches what the models in `models/` consume. It does not replace `PROJECT.md`.

## 1. What the models consume

All models in `models/` (linear classifier, ridge regression, 7-dimension ridge, Extra Trees, SVD-embedding calibrator) share the same data contract:

| Item | Requirement |
|---|---|
| Example | One **unordered pair of ORPHA IDs** (`ORPHA:x\|ORPHA:y`). |
| Inputs | Features computed from Orphanet, HPO, Open Targets and related tables. Seven set-valued families (phenotypes, genes, classifications, body systems, inheritance, onset, approved drugs) give similarity, overlap coefficient, size balance and availability, plus hashed shared/different term buckets. Prevalence gives its own features. |
| Target | A score in [0, 1] per paper. For the 7-dimension model this is `s_<dim>` for phenotype, genetic, mechanism, therapeutic, natural_history, diagnostic_confusability and comorbidity. |
| Aggregation | All papers for a pair are collapsed into one weighted mean. The summed paper weight `w_<dim>` is the ridge sample weight. |
| Splits | `splits.py` hashes the pair into train, validation or test. All papers for a pair stay together. |
| Rejections | A row is dropped if either disease has no ORPHA ID, if it maps to several IDs (unless expanded), if both sides are the same disease, if the score is outside [0, 1], or if a disease has no generated features. |

Consequences for scraping:

1. **Pairs are the unit of data, not papers.** The number of distinct usable pairs sets the effective sample size. Ten papers on one pair are one example with a better target.
2. **Targets are only as good as the per-pair evidence.** One abstract-based score is noisy. The weighted mean needs several independent papers per pair.
3. **Every pair needs ORPHA IDs on both sides and generated features.** Diseases without them are unusable, however good the paper.
4. **Each dimension needs enough pairs with weight above 0 in every split.** Otherwise `load_dimension_datasets` raises "No positive-weight ... scores".
5. **The model learns the relation between features and scores.** Pairs must therefore vary in their feature similarity, in both directions. A set where every pair is a clinical look-alike teaches nothing about separation.

## 2. Problems with paper-first scraping

Paper-first scraping means searching for comparison words and mapping whatever turns up. On the first 1,276 rows it produced:

- 15% of rows with no ORPHA ID. Names had to be mapped after the fact.
- 60% of pairs backed by a single paper.
- Very few papers with therapeutic (61), natural-history (65) or comorbidity (132) content.
- Almost no negatives (15 "unrelated" rows).
- Mostly "overlap", "coexisting" or "mimicking" case reports, so every pair is a clinical look-alike.
- Heavy hub concentration (ALS, MS, Parkinson, SSc, MSA).
- Only 6 distinct label values.

## 3. Recommended method: pair-first, feature-stratified sourcing

### Step 1. Define the disease universe from the features

- Run `download_databases.py` then `generate_features.py`, producing `.data/features/diseases.parquet`.
- Keep ORPHA diseases that are in that table and are disorders, not groups or categories. Exclude "Particular clinical situation" entries.
- Require at least one disease in each pair to be rare. Record rare/common status from the prevalence table.
- For common comparators (MS, Parkinson, Alzheimer, etc.), include them only if the feature table has them. Otherwise they cannot be model examples.
- Cap each disease at about 5% of the pairs so hubs cannot dominate.

### Step 2. Choose the pairs on purpose

Compute the model's own features for candidate pairs, and sample from strata so the target spans its full range:

| Stratum | Feature signature | Why the model needs it |
|---|---|---|
| A. Close relatives | High phenotype, gene and classification overlap | High-end anchors |
| B. Phenotype twins, different genes | High HPO overlap, no shared genes | Separates phenotype from genetic similarity |
| C. Genetic cousins | Shared genes, low phenotype overlap | The reverse of B |
| D. Shared treatment | Shared approved drug, different organ systems | Teaches the therapeutic dimension |
| E. Same class, unrelated biology | Same classification parent, nothing else shared | Hard negatives |
| F. Random cross-class | No shared features | Easy negatives |

Aim for roughly 300–400 pairs, with the strata balanced enough that no feature family is constant. Use a deterministic seed and record the stratum for each pair in the output, so results can be audited by stratum.

Hash-splitting means the final test set is about 15% of the pairs. Plan for at least 60 test pairs per dimension.

### Step 3. Search per pair, not per phrase

For each pair, build the query from Orphanet names and synonyms, so the ORPHA IDs are known in advance:

```
(TITLE_ABS:"<name A or synonym>" AND TITLE_ABS:"<name B or synonym>")
```

Use Europe PMC (`resultType=core`, plus `fullTextXML` for open-access PMC papers), the source `PROJECT.md` lists, so the earlier QC logic carries over.

- Run a general query first.
- Then run dimension-specific queries to fill the thin dimensions:
  - therapeutic: `treatment OR therapy OR response`
  - natural_history: `survival OR "age at onset" OR prevalence OR incidence`
  - comorbidity: `comorbid OR coexist OR association`
  - genetic: `gene OR mutation OR variant`
  - mechanism: `pathway OR pathogenesis`
- Keep the top 5 hits per query, then deduplicate by PMID.
- Reject papers that mention both diseases only incidentally, for example in a reference list or a long list of differentials. Require both names in the title or abstract, and the abstract must say something about how they relate.
- Include reviews, comparisons, cohorts, GWAS and case reports. Weight them by evidence strength, as `build_edge_ranking.py` does.

Target at least 3 papers per pair. A pair with fewer than 3 stays in the file but is flagged `low_evidence`.

### Step 4. Score each paper per dimension

Per paper and per pair, record the 7 weights (summing to 1) and the 7 scores.

- Prefer open-access full text. The abstract-versus-full-text comparison showed weights drift more than scores, and the dimension weights matter because they become the ridge sample weights.
- Use a written rubric for each dimension's 0–1 score, not a label-derived value. For example, genetic: 0 means no shared genes, 0.5 means related genes or pathway members, 1 means the same gene or the same variants.
- Fix the score anchors in the rubric (0, 0.25, 0.5, 0.75, 1) and allow values between them. This avoids the 6-value problem.
- Leave `s_<dim>` blank when the weight is 0, because the loader skips those rows.
- Record `evidence_checked` as abstract only, partial text or full text.

### Step 5. Add negatives and flag them

Stratum F pairs often have no papers that mention both diseases. That is itself informative, but it is not a paper-derived score.

- Include them as a separate file, `pseudo_negatives.csv`, with a constant low score and `source = no_cooccurrence`.
- Do not mix them into the paper-derived set unless you decide to. Evaluate on both separately.
- Pairs where a paper explicitly states the diseases are unrelated stay in the main file, as `relationship = unrelated`.

### Step 6. Output format

One row per (paper, pair), matching what `splits.py` expects:

`disease_a, disease_b, disease_a_orpha_id, disease_b_orpha_id, similarity_score, relationship, pmid, paper_title, year, link, open_access, evidence_checked, stratum, query_type, status_a, status_b`

with a companion file in the `paper_dimension_scores.csv` layout (`pmid, disease_a, disease_b, w_*, s_*`).

Keep the existing 1,276-row file as a separate source. Add the new file, rather than overwriting, so the two can be compared.

### Step 7. Check before training

Run `splits.py` and read the manifest. Accept the data only if:

- Rejections for missing ORPHA mapping or missing features are under 2%.
- The number of distinct pairs is at least 300, with at least 60% having 3 or more papers.
- Each dimension has at least 150 pairs with weight above 0, and at least 20 in each of train, validation and test.
- `similarity_score` has at least 20 distinct values, and no single value covers more than 20% of rows.
- No disease appears in more than 5% of pairs.
- Rare/rare and rare/common pairs are both well represented.

## 4. Leakage and bias to watch

- Orphanet-derived features and Orphanet-derived labels. A paper that only restates the Orphanet classification would inflate the apparent accuracy of the classification features. Prefer primary comparisons (cohorts, GWAS, head-to-head studies) over background reviews.
- Selection by feature similarity makes labels depend on the features by construction. This is intended for coverage, but report results per stratum so a good overall number cannot hide a failure inside one.
- Literature volume reflects research attention, not biology. Cap papers per pair so heavily studied pairs do not dominate the weights.
- The split is by pair, so do not tune the strata on test results.

## 5. Open questions

- The feature tables have not been generated in this checkout, so the exact disease universe and the number of usable rare-vs-common pairs are unknown until `generate_features.py` has run.
- Pseudo-negatives need a decision on score value and on whether they enter training.
