# Paper-input new-disease retrieval benchmark

## Question

Given only the text of a newly completed paper about a disease that is absent
from the trained graph, how well can v2 retrieve useful comparison diseases?
How does retrieval change as more of the paper becomes available?

This is an inductive retrieval benchmark, not a test of whether v2 can identify
the paper's focal disease by name.

## Frozen protocol

### Query diseases

- Start from the stable v2 **test-disease** split.
- Require at least one curated v2 relation and one lawfully acquired
  open-full-text paper containing an exact ORPHA mention in its title or
  abstract.
- Choose at most one paper per query disease using a deterministic rule:
  title mention first, then direct paper-pair evidence, then target-mention
  count, then body length, then stable paper ID.
- Use each paper for at most one query disease so disease-level results do not
  duplicate the same input document.
- The default cap is 200 query diseases. No result-dependent paper selection is
  allowed.

### Strict holdout

- All benchmark query diseases are removed together from:
  - fusion-model training pairs;
  - encoder catalogue vocabularies;
  - the retrieval gallery.
- Encoder statistics and fusion weights are fitted using v2 train diseases
  only.
- The benchmark paper text is introduced only after fitting.
- The source paper is never used to alter v2's training labels.

This is stricter than withholding one disease at a time. Other v2 validation
and test diseases may remain in the gallery, but no benchmark query disease
does.

### Information levels

For each selected paper, construct nested inputs:

1. `title`: title only;
2. `title_abstract`: title plus abstract;
3. `first_500_words`: title, abstract, then body in document order, truncated
   to 500 whitespace-delimited words;
4. `first_2000_words`: the same stream truncated to 2,000 words;
5. `full_text`: the stream truncated to 10,000 words.

The 10,000-word ceiling prevents a few very long papers from dominating
runtime. Word count used is stored for every query.

### Input variants

Run three preregistered variants:

- `masked_extracted` (primary): mask every detected ORPHA disease mention, then
  extract genes, HPO labels, drugs, inheritance, onset, and prevalence from the
  remaining text; also provide the masked text description.
- `masked_text_only`: the same masked text, with no structured extraction.
- `natural_extracted`: retain disease mentions and perform the same structured
  extraction. This estimates how much explicit disease naming helps.

Masking all disease mentions prevents trivial recovery when the paper directly
names a comparator. The target disease name and ORPHA ID are never supplied as
the new disease's name modality.

### Automated extraction

Use exact, boundary-aware matching against reference vocabularies already
loaded by v2:

- HGNC-style gene symbols from v2 knowledge;
- HPO preferred labels;
- ChEMBL drug preferred names;
- controlled inheritance and age-of-onset phrases;
- explicit prevalence expressions such as `1 in N` or percentages.

Extraction is deterministic and local. It does not use the held-out disease's
database profile, an LLM, or publisher-only text.

## Ground truth

Report two non-interchangeable targets.

### Primary: curated external relations

The union of v2's three independently constructed relation sets:

- Orphanet classification relation;
- shared curated causal gene;
- shared phase-2-or-later trial drug.

Only neighbours remaining in the retrieval gallery count. These labels are
independent of the benchmark paper text.

### Secondary: paper comparators

Other exact ORPHA diseases connected to the focal disease by the selected
paper's `paper_pairs` rows. This measures whether the model recovers
comparators discussed in the paper, but it is secondary because these pairs
were detected automatically and may be noisy.

## Metrics

Macro-average over query diseases:

- mean average precision (MAP);
- mean reciprocal rank (MRR);
- Hits@10;
- Recall@10 and Recall@50;
- nDCG@10.

Also report:

- number of queries and gallery size;
- median input words and extracted modality coverage;
- random-ranking baselines;
- an `oracle_profile` ceiling using the held-out disease's structured database
  record with its name and ontology hidden.

Primary comparisons are within the same fixed query cohort:

- performance versus information level;
- structured extraction versus text only;
- natural disease mentions versus all mentions masked.

No hyperparameter is selected on benchmark results.

## Interpretation

Success means performance rises as more paper information is provided and
beats random retrieval on curated relations. It does not establish clinical
validity. Important limitations are:

- paper selection requires open full text and therefore has access bias;
- curated relation labels cover only three relationship types;
- automated entity extraction and paper comparators are imperfect;
- database descriptions may ultimately cite some of the same scientific
  literature, even though the benchmark papers are not training inputs;
- one paper may not describe every defining feature of a disease.
