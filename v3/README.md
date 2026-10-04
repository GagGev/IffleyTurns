# Rare-disease similarity graph, v3

v3 asks a harder question than v2. Instead of checking that the similarity graph reproduces database relations it was
built next to, it checks that the graph **anticipated regulatory decisions made after its training cutoff**. Two rare
diseases count as related on the date the same drug first holds FDA or EMA orphan designations for both. Models trained
on knowledge before 2018 are scored on the 1,123 relations that regulators established from 2018 to September 2026.

Inputs are `.data/features/` and `.data/databases/` only (including `databases/regulatory/`). v3 does not read
`literature_review/`, `.data/literature_acquisition/`, `.data/splits/` or `.data/models/`.

## Run

```bash
python v3/regulatory.py        # ~1 min: parse FDA + EMA designations, map them to Orphanet, write dated relations
python v3/run_evaluation.py    # ~30 min (GPU): validation + test folds, report, figure, production stacker choice
python v3/build_graph.py       # fit on everything up to the end of the data, write the graph and figures
python v3/place_disease.py --json v3/examples/new_disease_example.json        # place a new disease
python v3/place_disease.py --json v3/examples/new_disease_example.json --add  # ... and insert it into the graph
python v3/place_disease.py --orpha ORPHA:558 --hide ontology,name             # re-place a known disease as if new
python -m pytest v3/tests -q
```

The neural model uses PyTorch with CUDA if available (tested with torch 2.11 + cu128 on an RTX 5060 Laptop GPU); it
falls back to CPU. Outputs go to `v3/.data/`: `regulatory/` (designations, relations, mapping audit), `evaluation/`
(`report.md`, `summary.json`, per-query metrics, `map_by_model.png`), `graph/`, `model/`.

## Ground truth: dated orphan-designation relations

**Why orphan designations.** A designation is granted only after the regulator accepts the sponsor's scientific
case that the drug is plausible for that rare disease. The FDA requires a "scientific rationale" with supporting data
(21 CFR 316.20), and the EMA's COMP reviews "medical plausibility" (Regulation (EC) No 141/2000). When one molecule
is accepted for two diseases in separate decisions, an independent expert body has judged that the two diseases share
a treatable mechanism. That is the relation a similarity graph for drug repurposing should anticipate. Unlike curated
classifications, each relation has a date, so prediction can be tested strictly forward in time.

**Why it is hard to obtain.** The relations do not exist in any database; `regulatory.py` builds them in four steps:

1. Parse 11,124 designations: 7,902 FDA (since 1983) and 3,222 EMA (since 2000; refusals dropped).
2. Identify each drug as its ChEMBL parent molecule through Open Targets names, synonyms and trade names, with salt
   forms stripped; otherwise use a normalized name key. 6,687 drugs, 45% resolved to ChEMBL.
3. Map the free-text indication ("Treatment of ...") to Orphanet with a tiered matcher. It uses Orphanet names and
   synonyms plus Mondo exact synonyms of Orphanet-equivalent terms, and handles purpose phrases, qualifiers ("... in
   patients with X"), lists ("X and Y", where every piece must map), negations ("non-small cell") and optional
   modifiers ("proximal", "classic"). 8,329 designations (75%) are mapped:

   | Tier | Designations | Audited precision |
   |---|---|---|
   | exact | 6,543 | ~24/25 |
   | contained | 1,428 | ~22/25 |
   | qualifier | 302 | 20/20 |
   | list | 56 | 12/12 |

   The audit sample is in `mapping_audit_sample.csv` and the unmapped wording in `unmapped_indications.csv`.
4. Pair the diseases. A pair needs two distinct designation records, so one joint record never relates its own
   diseases. Pairs where one node is the other's Orphanet ancestor are skipped. The 4 broad drugs designated for more
   than 12 nodes, such as checkpoint inhibitors, create no relations. The pair's date is the first date both
   designations were held.

This yields 2,473 dated relations among 585 nodes. 306 are "approved both", meaning both designations led to an
approved product. Their number grows from about 10 a year in the 1990s to 100–190 a year since 2014.

**Nodes.** There are 9,525 nodes: the 7,493 Orphanet diseases from v2, plus all 2,032 Orphanet groups with 2–200
member diseases. All eligible groups are included, not only the 158 that were designated. Otherwise being a node would
itself reveal that a group had been designated.

## Protocol

| Fold | Training knowledge | Relations to predict | Queries |
|---|---|---|---|
| validation | before 2014-01-01 (818 relations) | 2014–2017 (532) | 303 |
| test | before 2018-01-01 (1,350 relations) | 2018-01 to 2026-09 (1,123) | 447 |

Leakage controls:
- **Drug-free static profiles.** Each node has nine modalities: phenotype, gene, pathway, ontology, name, text,
  inheritance, onset and prevalence. Unlike v2, it has no drug, drug-target or Open Targets association modality.
  2,086 designated-drug words are removed from the descriptions, for example "imatinib". Biomedical words that are
  also drug names, such as "glutamine" and "factor", are kept.
- **Time-sliced history.** Everything date-dependent is computed from a snapshot at the cutoff: designated drugs,
  their targets and target pathways, the known relation graph, designation counts and years. Models, the stacker's
  training origins and early stopping all use data before the cutoff only.
- **Candidates.** Each query is a node that gains a relation in the window. It ranks all 9,525 nodes, excluding
  itself, nodes it was already related to, and its Orphanet ancestors and descendants. Galleries:
  - `full`: every node;
  - `warm`: only nodes designated before the cutoff, which removes the "has this disease ever been designated"
    signal;
  - `full_approved`: positives restricted to approved-both relations.
- **Metrics.** Per-query MAP (primary), AUROC, MRR, Hits@10, nDCG@10 and Recall@50, with bootstrap 95% CIs and paired
  one-sided Wilcoxon tests. Also reported: strata (warm/cold, group/disease, oncology) and the global precision@K of
  new pairs among designated diseases.

The production stacker variant is chosen on validation MAP only; the test fold is reported once.

## Models

**Static similarity (drug-free, works for any new disease).**
- `v2_shipped`: v2's production logistic fusion, with v2's drug modalities removed.
- `v2_retrained`: the same architecture refit on regulatory relations: non-negative logistic fusion of the nine
  cosines plus availability terms.
- `gbm_static`: gradient boosting on the same inputs.
- `v3_neural`: a multi-task neural network, run as a 3-seed ensemble on the GPU:
  - each modality is projected from its sparse vector to 64 dimensions, then attention pooling with a null token
    gives a 128-d disease embedding;
  - modality dropout is 0.2;
  - a symmetric pair MLP reads [z_a ⊙ z_b, |z_a − z_b|, cosines, availability], plus a non-negative "wide" linear
    term.
  
  It trains on three tasks: regulatory relations before the cutoff, Orphanet siblings (with ontology and name
  hidden), and shared causal genes (with gene and pathway hidden). The embedding learns only from the two auxiliary
  tasks; the therapeutic head learns on top of it. Training the encoder on the 800–1,400 therapeutic pairs made it
  memorize which diseases attract drugs.
- `v3_static`: the standardized sum of `v3_neural` and `v2_retrained`. This is the graph's similarity score.

**Forecast (`v3_stacker` and ablations).** Gradient boosting over five feature families:
- **static:** the three static logits, nine cosines, and the best static score to each disease's known relatives;
- **history:** designation counts, years since the first designation, relation degree, drugs already shared;
- **mechanism:** target and target-pathway overlap of designated drugs, and one disease's genes or pathways
  targeted by the other's drugs;
- **graph:** common neighbours, Adamic–Adar and Jaccard over known relations;
- **node:** group/disease, group size, oncology, same top-level category.

It is trained on rolling origins 2, 4, 6 and 8 years before the cutoff. At each origin, the features come from that
origin's snapshot and a static model trained before it, and the labels are the relations formed between that origin
and the cutoff. Negatives are sampled separately among warm and cold nodes and reweighted to the true mix.

**Reference baselines:** random; degree, which ranks by designation count and relation degree; Adamic–Adar on the known
relation graph; drug-mechanism overlap; and the v1 weighted Jaccard without its drug annotations.

## Results

![MAP by model](.data/evaluation/map_by_model.png)

Test fold (knowledge before 2018, relations from 2018 to 2026-09), all 9,525 nodes as candidates, 447 queries:

| Model | MAP [95% CI] | AUROC | Hits@10 | Recall@50 |
|---|---|---|---|---|
| random | 0.001 | 0.497 | 0.000 | 0.003 |
| v1 weighted Jaccard | 0.013 [0.010, 0.017] | 0.767 | 0.065 | 0.088 |
| drug mechanism | 0.040 [0.029, 0.054] | 0.674 | 0.159 | 0.136 |
| Adamic–Adar | 0.051 [0.041, 0.064] | 0.597 | 0.251 | 0.164 |
| degree | 0.070 [0.060, 0.080] | 0.899 | 0.385 | 0.295 |
| v2 shipped | 0.057 [0.045, 0.069] | 0.807 | 0.233 | 0.216 |
| v2 retrained | 0.056 [0.046, 0.067] | 0.841 | 0.262 | 0.250 |
| v3 neural | 0.062 [0.051, 0.073] | 0.859 | 0.284 | 0.264 |
| v3 static (graph similarity) | 0.060 [0.049, 0.071] | 0.860 | 0.289 | 0.265 |
| v3 stacker, all features | 0.113 [0.098, 0.130] | 0.936 | 0.432 | 0.394 |
| **v3 stacker without history (production)** | **0.109 [0.093, 0.126]** | **0.921** | **0.405** | **0.378** |

- **The production forecast beats v2 on future relations.** Its test MAP is +0.052 higher than v2 shipped
  [+0.037, +0.068] (p = 1e-24) and +0.053 higher than v2 retrained. It also beats the degree baseline by +0.039
  [+0.021, +0.058]. That baseline is hard to beat: "diseases that already attract drugs attract more" reaches AUROC
  0.90. The "no history" variant won validation narrowly (0.085 vs 0.080, not significant) and ranks pairs without
  using designation counts.
- **The static similarity improves modestly.** v3 static beats v2 retrained on the test fold: MAP +0.004
  [+0.001, +0.006], AUROC +0.019, both p < 1e-14. On validation it trails on MAP (−0.006) but leads on AUROC (+0.017).
  Against v2 shipped, the mean MAP gain is +0.003 [−0.003, +0.009], an interval that includes zero, while the AUROC
  gain is +0.053. In the warm gallery, where every candidate was already designated, v3 static leads v2 shipped by
  0.148 vs 0.134. In other words, the multi-task network alone does not improve much on the logistic fusion. The large
  gain comes from combining static similarity with time-sliced regulatory evidence.
- **Static similarity matters most inside the stacker.** Removing it drops test MAP from 0.113 to 0.078, and
  permutation importance ranks it first (AUC drop 0.28, versus 0.04 for history).
- **Global precision.** Among the 201,423 pairs of already-designated diseases, 0.42% become related after 2018.
  The production model's top 100 pairs have 27% precision (about 65× the base rate), and its top 500 have 16%.
- **Strata.** On test queries never designated before the cutoff ("cold", 94), static similarity is as good as the
  stacker (0.084–0.089 MAP). History and graph features help most for oncology (stacker 0.156 vs static 0.042) and
  for Orphanet groups.
- **Approved-both relations** (46 test queries): MAP is 0.144 for the production model and 0.171 for the full
  stacker, versus 0.046 for v2 shipped. The intervals are wide.

The full tables (all galleries, both folds, strata, paired tests) are in `.data/evaluation/report.md`.

## The graph

GRAPH_SECTION

## Placing a new disease

Describe the disease in JSON; every field is optional except `name`. The fields match v2, plus `oncology`:

```json
{
  "id": "USER:my-disease", "name": "...", "synonyms": ["..."], "description": "free text",
  "phenotypes": {"HP:0001250": 1.0, "HP:0001263": 0.8}, "genes": ["CDKL5"],
  "drugs": ["CHEMBL1234 or a drug name"], "inheritance": ["X-linked dominant"], "onset": ["Infancy"],
  "prevalence": 1e-6, "ontology_parents": ["ORPHA:102369"], "oncology": false
}
```

`drugs` are treated as orphan-designated today. They feed only the forecast, through shared targets and genes
targeted by the other disease's drugs; the similarity stays drug-free. Neighbours are ranked by similarity
(`--rank-by forecast` is available). Each neighbour is listed with:
- its similarity and forecast percentiles against random pairs;
- the modalities behind the score: linear contribution, neural occlusion (the drop when the modality is withheld from
  both diseases), and the shared items;
- regulatory evidence: shared drug targets, and genes targeted by the other disease's drugs.

`--add` appends the disease and its edges to `nodes.csv`, `edges.csv` and the GraphML, and records it in
`user_diseases.jsonl`. Later placements are compared against it as well.

PLACEMENT_EXAMPLE

## Limitations

- **Static annotations are a 2026 snapshot.** Orphanet phenotypes, genes and descriptions carry no dates, so a
  description written after 2018 could reflect knowledge the 2018 model would not have had. Drug names are removed
  from the text, and genes and phenotypes are not drug data. Still, the static inputs are not strictly time-sliced.
  v2 shipped's weights were also fitted on 2026 data, which can only favour v2 in the comparison.
- **The relations are a lower bound.** An unrelated pair may simply not have been tried; sponsors' commercial
  choices shape designations, and oncology is over-represented. A quarter of designations stay unmapped, and the
  "contained" tier has about 10% mapping errors, usually a too-coarse group.
- **Drug identity is approximate.** 45% of drugs resolve to ChEMBL; the rest match by normalized name, so two names
  of one molecule can split one drug into two, losing relations.
- **The absolute numbers are modest.** For 40% of query diseases, a future relation appears in the top 10 of 9,525
  candidates. That is far above chance and v2, but most top-ranked pairs have not, or not yet, been linked by a
  regulator. The graph's edges are hypotheses to review, not findings.
