# Scientific Literature Acquisition and Annotation Protocol for Rare-Disease Similarity

## 1. Purpose

This protocol defines a reproducible method for discovering scientific papers,
retrieving legally accessible text, extracting disease-pair evidence, and
constructing defensible labels for rare-disease similarity models.

The primary output is not a single subjective similarity number. It is a
versioned evidence dataset containing:

1. paper-level observations;
2. dimension-specific ordinal scores with explicit anchors;
3. evidence quality and extraction confidence stored separately from the score;
4. explicit related, distinct, unrelated, and unknown controls;
5. source passages and provenance sufficient for independent review; and
6. leakage-safe train, validation, and test assignments.

This protocol is intended for research and model evaluation. It does not
produce clinical diagnoses or treatment recommendations.

---

## 2. Problems this protocol addresses

The existing literature dataset and experiments revealed several systematic
problems:

- Overall scores largely encode the categories `related but distinct`,
  `similar`, and `unrelated`, rather than a continuously measured biological
  similarity.
- Almost all labels were derived from abstracts; full-text context, cohort
  definitions, limitations, and negative findings were usually unavailable.
- Approximately 60% of disease pairs had only one contributing paper.
- Verified unrelated controls were extremely rare. Absence of a paper was at
  risk of being confused with evidence of unrelatedness.
- Dimension coverage was highly uneven. Therapeutic, natural-history, and
  comorbidity targets were too sparse for reliable held-out evaluation.
- The existing `w_*` values sum to one per paper, so they behave like
  compositional relevance weights rather than independent evidence confidence.
- The same paper can contribute different pairs across different splits,
  allowing paper-specific annotation style or findings to cross split
  boundaries.
- Feature improvements modestly improved global ranking but did not consistently
  improve dimension-specific prediction. This indicates that better features
  cannot compensate for weak or mismatched labels.

The protocol below treats label construction as a measurement problem rather
than a scraping-volume problem.

---

## 3. Preregister the review before collecting papers

Create a versioned protocol record before running any query. It should contain:

- review title and protocol version;
- date range covered;
- disease universe and Orphanet release;
- databases and API versions;
- inclusion and exclusion criteria;
- query templates;
- score definitions;
- primary and secondary outcomes;
- planned control construction;
- aggregation procedure;
- split strategy;
- minimum evidence requirements; and
- planned sensitivity analyses.

Any change after inspecting results must be recorded as a protocol amendment.
Do not silently alter score anchors, inclusion rules, or controls to improve
model performance.

---

## 4. Define the disease universe

### 4.1 Canonical identifiers

Use one canonical ORPHA identifier for each disease entity. Preserve all source
identifiers separately:

- ORPHA;
- MONDO;
- OMIM;
- MeSH;
- UMLS;
- ICD-10/ICD-11; and
- source-specific identifiers.

Never replace an ambiguous mapping with a guessed identifier.

### 4.2 Entity granularity

Record the Orphanet entity type:

- disease;
- group of disorders;
- clinical subtype;
- etiological subtype;
- malformation syndrome;
- category; or
- other.

Pairs should normally compare entities at compatible granularity. A broad
disease group versus a narrow molecular subtype must be flagged as a
granularity mismatch and excluded from primary analysis unless that comparison
is the explicit subject of the paper.

### 4.3 Synonym dictionary

Build a versioned search dictionary for every canonical disease:

- preferred name;
- exact synonyms;
- historical names;
- eponyms;
- abbreviations;
- gene-associated names;
- spelling variants; and
- explicit exclusions for ambiguous acronyms.

Every synonym must retain its source and mapping confidence. Short ambiguous
abbreviations should not be searched without contextual terms.

---

## 5. Literature sources and lawful acquisition

Use documented APIs and text-and-data-mining routes rather than unrestricted
publisher-page scraping.

### 5.1 Discovery sources

Recommended discovery order:

1. **Europe PMC** for biomedical metadata, abstracts, references, citations,
   and open-access full text.
2. **PubMed/NCBI E-utilities** for authoritative PubMed metadata and publication
   types.
3. **Crossref** for DOI normalization and bibliographic deduplication.
4. **PMC Open Access** for legally reusable full-text XML.
5. **Unpaywall or publisher TDM endpoints** for open-access resolution when
   permitted by their terms.

Record the endpoint, query, retrieval timestamp, response identifier, license,
and checksum for every acquired document.

### 5.2 Access classes

Assign one of these values:

- `open_full_text_xml`;
- `open_full_text_pdf`;
- `licensed_tdm_full_text`;
- `abstract_only`;
- `metadata_only`;
- `unavailable`; or
- `retracted_or_removed`.

Do not bypass paywalls, authentication, robots rules, rate limits, or publisher
terms. A paper unavailable for lawful full-text extraction may remain in the
discovery log, but abstract-derived labels must be marked and analyzed
separately.

### 5.3 Polite and reproducible requests

- Identify the application and provide a contact address where required.
- Respect published request-rate limits.
- Retry transient failures with exponential backoff and jitter.
- Cache immutable responses.
- Use conditional requests when supported.
- Never launch parallel traffic that exceeds the provider's policy.
- Store raw responses before parsing.
- Record failed requests instead of silently dropping them.

---

## 6. Search strategy

Pair-co-occurrence searches alone create severe selection bias: they find pairs
already suspected to be related. Use three complementary search tracks.

### 6.1 Track A: direct disease-pair evidence

For diseases `A` and `B`, search:

```text
("A preferred name" OR "A synonym 1" OR ORPHA identifier)
AND
("B preferred name" OR "B synonym 1" OR ORPHA identifier)
```

Run an unqualified query first. Then run dimension-specific queries:

- phenotype: `phenotyp*`, `clinical feature*`, `manifestation*`, `symptom*`;
- genetics: `gene*`, `variant*`, `inherit*`, `genotype*`;
- mechanism: `pathway*`, `mechanis*`, `molecular`, `pathogenesis`;
- therapeutic: `treatment*`, `therap*`, `drug*`, `intervention*`;
- natural history: `natural history`, `progression`, `onset`, `survival`;
- diagnostic confusability: `differential diagnosis`, `mimic*`,
  `misdiagnos*`, `distinguish*`;
- comorbidity: `comorbid*`, `co-occur*`, `association`, `risk of`.

Do not require a dimension term in the only query; doing so can miss relevant
papers that use unexpected terminology.

### 6.2 Track B: disease-centred evidence

Search each disease independently with dimension terms, then detect mentions of
other canonical diseases in title, abstract, keywords, full text, tables, and
supplements. This discovers relationships that are not visible in pairwise
titles or abstracts.

### 6.3 Track C: control and disconfirming evidence

For each candidate related pair, deliberately search for:

- distinguishing features;
- absence of shared mechanisms;
- negative association studies;
- differential diagnosis;
- false-positive historical associations;
- reclassification;
- non-replication; and
- explicit statements that two entities are unrelated or mechanistically
  distinct.

Controls must not be created solely because no paper was found.

### 6.4 Citation expansion

For included reviews and high-quality primary studies:

- inspect references;
- inspect citing papers;
- inspect related-article recommendations; and
- record whether a paper was query-discovered or citation-discovered.

Apply the same screening criteria to citation-expanded papers.

### 6.5 Search stopping rule

For each disease pair, stop only when all preregistered sources and query
templates have been run and one of the following holds:

- the planned date range is exhausted;
- two consecutive citation-expansion rounds yield no eligible new paper; or
- the predefined evidence saturation criterion is met.

Record the stopping reason.

---

## 7. Deduplication and publication families

Deduplicate in this order:

1. PMID;
2. PMCID;
3. normalized DOI;
4. trial or registry identifier plus publication role;
5. normalized title, year, and first author; and
6. manual review of likely duplicates.

Link papers belonging to the same underlying study:

- conference abstract and full paper;
- preprint and peer-reviewed article;
- interim and final analysis;
- subgroup and primary report;
- correction or retraction; and
- secondary analysis of the same cohort.

Do not count multiple reports of the same participants as independent
replication. Assign a `study_family_id`.

---

## 8. Screening procedure

### 8.1 Title and abstract screening

Two independent reviewers should classify each record:

- `include`;
- `exclude`;
- `uncertain`; or
- `background_only`.

At least one `include` or `uncertain` vote advances a record to full-text
screening.

### 8.2 Full-text inclusion criteria

Include a paper-level disease-pair observation when:

- both diseases map unambiguously to canonical entities;
- the paper provides direct comparative, mechanistic, epidemiological,
  diagnostic, or therapeutic evidence connecting or distinguishing them;
- the evidence can be assigned to at least one predefined dimension; and
- the relevant population, model system, or evidence type is identifiable.

### 8.3 Exclusion criteria

Exclude or separately flag:

- healthy-control comparisons represented as disease-disease controls;
- keyword co-occurrence without a substantive relationship;
- broad categories mapped to a specific disease without justification;
- duplicate reports from the same study family;
- animal or cell evidence presented as clinical similarity;
- case reports used to infer population-level comorbidity;
- narrative assertions with no cited evidence;
- withdrawn or retracted findings, except in a retraction audit; and
- papers where the disease mapping remains ambiguous after adjudication.

Every exclusion must have a controlled reason code.

### 8.4 Inter-reviewer reliability

During a pilot phase, independently screen at least 100 records or 10% of the
first batch, whichever is larger. Report:

- raw agreement;
- Cohen's kappa for two reviewers or Fleiss' kappa for more reviewers;
- disagreement types; and
- changes made to the instructions.

Continue calibration until the preregistered agreement threshold is reached.

---

## 9. Evidence extraction

Each included paper must be extracted independently by two reviewers for the
primary test set. Training data may use one extractor plus audited verification
after the extraction process demonstrates adequate reliability.

### 9.1 Required provenance

Store:

- PMID, PMCID, DOI, and study-family identifier;
- title, year, journal, and publication type;
- query or citation route;
- access class and license;
- canonical ORPHA pair;
- exact disease names used by the paper;
- entity granularity;
- study population and setting;
- sample size per disease;
- study design;
- model system: human, animal, cell, or computational;
- full-text section, page/table/figure, and source passage;
- extractor and reviewer identifiers;
- extraction timestamp; and
- protocol and rubric version.

### 9.2 Evidence passages

Every non-missing dimension score must cite at least one source passage.
Passages should include enough surrounding context to preserve:

- direction of the result;
- population or model system;
- comparator;
- uncertainty;
- negation;
- subgroup qualification; and
- whether the statement is a result or background assertion.

Store short quotations only where legally permitted. Otherwise store a
structured paraphrase plus a stable locator.

### 9.3 Keep claims atomic

One extraction row should represent one paper, one canonical disease pair, one
dimension, one population/model system, and one direction of evidence. Split
rows when a paper reports incompatible populations, subtypes, or evidence
directions.

---

## 10. Dimension scoring rubric

Use a five-level ordinal score and preserve the ordinal value. A normalized
value `score / 4` may be derived for models, but it must not replace the source
ordinal label.

### 10.1 Universal score meaning

- **0 — none/opposed:** explicit evidence of no meaningful similarity in this
  dimension, or evidence that mechanisms/features are fundamentally different.
- **1 — weak:** broad or incidental overlap without shared defining features.
- **2 — moderate:** reproducible partial overlap with important differences.
- **3 — strong:** substantial shared defining evidence; differences remain.
- **4 — very strong:** near-equivalent or directly shared evidence in this
  dimension.
- **NA — not assessed:** the paper does not provide evidence for this dimension.

`NA` is not zero.

### 10.2 Phenotype anchors

- 0: no shared core manifestations or explicitly non-overlapping phenotype.
- 1: same broad organ system only.
- 2: several shared manifestations, but core phenotype remains distinguishable.
- 3: substantial core overlap or frequent clinical mimicry.
- 4: near-indistinguishable clinical phenotype in the studied context.

### 10.3 Genetic anchors

- 0: explicit evidence of unrelated genetic causes.
- 1: broad inheritance or chromosome-level similarity only.
- 2: shared gene family, locus, inheritance pattern, or partially overlapping
  genetic architecture.
- 3: shared causal pathway or reproducible overlap in causal genes.
- 4: same principal causal gene/variant class with closely related disease
  consequences.

The same gene must not automatically receive a score of 4; phenotypically
divergent allelic disorders require evidence-based scoring.

### 10.4 Mechanism anchors

- 0: incompatible or explicitly distinct mechanisms.
- 1: same broad biological process.
- 2: interacting or partially overlapping pathways.
- 3: shared central pathway, molecular defect, or causal process.
- 4: essentially the same demonstrated causal mechanism.

### 10.5 Therapeutic anchors

- 0: incompatible treatment strategy or evidence that the treatment does not
  transfer.
- 1: same broad care category only.
- 2: shared symptomatic management or target class.
- 3: shared effective intervention or target with disease-specific differences.
- 4: same evidence-supported disease-modifying intervention and target context.

### 10.6 Natural-history anchors

- 0: clearly incompatible onset, progression, or prognosis.
- 1: one broad temporal feature in common.
- 2: partial similarity in onset or progression.
- 3: substantially similar disease course.
- 4: near-equivalent onset, progression, complications, and prognosis.

### 10.7 Diagnostic-confusability anchors

- 0: readily distinguished with no documented confusion.
- 1: superficial resemblance.
- 2: occasional differential-diagnosis overlap.
- 3: frequent mimicry requiring targeted testing.
- 4: routinely indistinguishable without molecular or highly specific testing.

### 10.8 Comorbidity anchors

- 0: evidence against co-occurrence beyond expectation.
- 1: isolated reports without population support.
- 2: suggestive association or replicated case-series evidence.
- 3: robust epidemiological association with appropriate controls.
- 4: strong replicated association with temporal and mechanistic support.

Co-mention, referral bias, and a shared diagnostic work-up are not sufficient
for a high comorbidity score.

---

## 11. Separate score, relevance, quality, and confidence

Do not use one field for all four concepts.

### 11.1 Dimension applicability

For each dimension:

- 0: not assessed;
- 1: indirect/background evidence;
- 2: directly assessed.

Applicability values are independent and do not need to sum to one.

### 11.2 Evidence quality

Record components separately:

- study design;
- sample size and precision;
- directness to the disease pair and dimension;
- human versus model-system evidence;
- independent replication;
- risk of bias;
- full-text availability;
- preregistration;
- correction/retraction status; and
- overlap with other study cohorts.

Derive a quality tier only after storing the components:

- `high`;
- `moderate`;
- `low`; or
- `very_low`.

Quality affects confidence in a score, not the biological direction or
magnitude of the score itself.

### 11.3 Extraction confidence

- 1: ambiguous or abstract-only;
- 2: full-text evidence with some interpretive uncertainty;
- 3: direct, clearly located full-text evidence.

### 11.4 Reviewer agreement

Store both reviewers' original scores before adjudication. Report:

- weighted Cohen's kappa for ordinal scores;
- intraclass correlation for normalized scores;
- percentage agreement within one ordinal level; and
- dimension-specific disagreement rates.

---

## 12. Constructing controls correctly

Use three distinct control types.

### 12.1 Verified unrelated controls

Require explicit comparative or mechanistic evidence supporting unrelatedness
or meaningful non-overlap. These are the strongest controls.

### 12.2 Related-but-distinct hard controls

These pairs share an organ system, broad phenotype, or diagnostic context but
have important mechanistic or clinical differences. They test fine-grained
ranking and must not be labelled simply `negative`.

### 12.3 Unobserved matched controls

Sample from the disease universe while matching:

- disorder granularity;
- broad body system;
- annotation coverage;
- prevalence class where possible; and
- publication volume.

Exclude all known evidence edges. Label these pairs `unobserved`, not
`unrelated`. They may be used for retrieval stress tests, not supervised
clinical ground truth.

### 12.4 Minimum control balance

For every disease used in primary evaluation, aim for:

- at least one supported related pair;
- at least one related-but-distinct pair; and
- at least one verified or adjudicated unrelated control.

If verified controls cannot be obtained, report that limitation rather than
substituting unobserved pairs.

---

## 13. Adjudication and consensus

1. Preserve both independent extractions.
2. Automatically flag:
   - score disagreement greater than one level;
   - conflicting disease mappings;
   - opposite evidence directions;
   - abstract/full-text disagreement;
   - study-family duplicates; and
   - score without a source locator.
3. A third reviewer adjudicates flagged rows without seeing model predictions.
4. Record the adjudicated value, reason, and reviewer.
5. Never overwrite original reviewer values.

Model errors must not be used to retroactively change labels unless independent
source review identifies a genuine annotation error.

---

## 14. Pair-level aggregation

Keep paper-level observations as the source of truth.

For an initial benchmark:

- aggregate within a study family first, so one cohort receives one vote;
- use the weighted median ordinal score as the primary pair score;
- use evidence quality and extraction confidence as preregistered weights;
- cap each study family's maximum contribution;
- report the unweighted median as a sensitivity analysis;
- report the number of independent study families;
- report score range and median absolute deviation; and
- mark pairs with substantial disagreement as heterogeneous.

For a larger dataset, use a hierarchical ordinal model with paper/study-family
random effects rather than treating decimal scores as noise-free continuous
measurements.

Do not combine dimensions into one overall target unless the intended use
defines and preregisters the aggregation weights. Preserve the seven-dimensional
vector as the primary representation.

---

## 15. Leakage-safe dataset construction

Publish multiple evaluation regimes rather than one optimistic random split.

### 15.1 Paper- and pair-blocked split

Create a graph whose nodes are:

- canonical disease pairs;
- paper identifiers; and
- study-family identifiers.

Connect each paper/study family to every pair it annotates. Assign connected
components to exactly one split. This prevents the same paper, cohort, or
duplicate pair from spanning train and test.

### 15.2 Disease-held-out split

Create a second test set in which neither disease appears in training. This
measures generalization to newly represented diseases.

### 15.3 Time-based split

Train on papers published before a preregistered date and test on later papers.
This estimates whether the method anticipates subsequently reported
relationships.

### 15.4 Stratification

Balance, where feasible:

- dimension;
- score level;
- evidence quality;
- rare-rare versus rare-common comparisons;
- disease granularity;
- publication year; and
- number of papers per pair.

The test set must remain untouched until the protocol, features, model family,
and validation procedure are fixed.

---

## 16. Evaluation

### 16.1 Primary metrics

For each sufficiently populated dimension:

- weighted and unweighted MAE;
- RMSE;
- Spearman correlation;
- ordinal weighted kappa;
- calibration by score level;
- Precision@K;
- mean reciprocal rank; and
- nDCG for disease-centred retrieval.

### 16.2 Required baselines

- train-set mean or median;
- exact Jaccard;
- ontology-aware semantic similarity;
- current heuristic;
- current production model; and
- publication-volume baseline.

The publication-volume baseline detects whether the model merely predicts
well-studied pairs.

### 16.3 Uncertainty

Report pair-level bootstrap confidence intervals. For disease-centred ranking,
resample query diseases rather than individual candidate edges.

Do not make primary claims for a dimension with:

- fewer than 100 held-out pair labels;
- fewer than 50 examples in a required score/control group; or
- insufficient effective sample size after study-family weighting.

Smaller results may be reported as exploratory with explicit uncertainty.
Final thresholds should be justified by a prospective power analysis.

### 16.4 Sensitivity analyses

At minimum:

- full text only versus abstract-inclusive;
- high/moderate quality only;
- multi-paper pairs only;
- paper-blocked;
- disease-held-out;
- rare-rare pairs only;
- compatible entity granularity only;
- weighted versus unweighted aggregation; and
- removal of each high-degree disease hub.

---

## 17. Recommended data schema

### 17.1 `papers`

```text
paper_id
pmid
pmcid
doi
study_family_id
title
year
journal
publication_type
access_class
license
is_retracted
retrieved_at
raw_document_sha256
discovery_route
query_id
```

### 17.2 `disease_pair_observations`

```text
observation_id
paper_id
study_family_id
orpha_id_a
orpha_id_b
entity_type_a
entity_type_b
population
model_system
study_design
sample_size_a
sample_size_b
dimension
ordinal_score
dimension_applicability
evidence_direction
quality_tier
extraction_confidence
source_section
source_locator
source_passage_or_paraphrase
reviewer_id
rubric_version
created_at
```

### 17.3 `adjudications`

```text
observation_group_id
reviewer_1_score
reviewer_2_score
disagreement_type
adjudicated_score
adjudicator_id
adjudication_reason
adjudicated_at
```

### 17.4 `pair_consensus`

```text
pair_id
dimension
consensus_ordinal_score
normalized_score
score_interval_low
score_interval_high
independent_study_families
full_text_studies
quality_tier
heterogeneity_flag
control_type
protocol_version
```

### 17.5 `search_log`

```text
query_id
database
query_text
protocol_version
requested_at
completed_at
result_count
response_sha256
status
error
```

---

## 18. Automated quality gates

Reject a release if any of the following occurs:

- a scored observation lacks a source locator;
- an `NA` dimension is converted to zero;
- an ambiguous ORPHA mapping enters the primary dataset;
- duplicate study families are counted as independent evidence;
- a paper or study family spans train and test in the paper-blocked split;
- a healthy control is represented as a disease;
- an unobserved pair is labelled verified unrelated;
- a retracted paper contributes without an explicit retraction flag;
- score anchors or aggregation rules differ from the registered version; or
- raw source checksums and query logs are missing.

Produce a release manifest containing row counts, rejection counts, mapping
coverage, full-text coverage, dimension coverage, reviewer agreement, score
distribution, control distribution, and split-integrity checks.

---

## 19. Recommended implementation phases

### Phase 1: rubric calibration

- Select 30-50 diverse pairs.
- Acquire full text where lawful.
- Have two domain-informed reviewers annotate every dimension.
- Refine examples and instructions before model training.

### Phase 2: balanced benchmark

- Construct a preregistered evaluation set first.
- Prioritize verified controls and hard related-but-distinct cases.
- Require dual extraction and adjudication.
- Freeze this benchmark before expanding training data.

### Phase 3: training expansion

- Run systematic searches over the full disease universe.
- Use active learning only to prioritize review, never to assign labels.
- Maintain balanced dimension and score coverage.

### Phase 4: prospective evaluation

- Freeze the model.
- Collect newly published papers after a cutoff date.
- Evaluate without revising the historical test set.

---

## 20. Success criteria

The acquisition and annotation process is successful when:

- full-text evidence supports the majority of primary labels;
- reviewer agreement reaches the preregistered threshold;
- every score is traceable to a source passage or stable locator;
- related, distinct, unrelated, and unknown states are not conflated;
- each primary dimension has adequate independent test evidence;
- paper-, study-, and disease-level leakage checks pass;
- known relationships rank above verified controls with confidence intervals;
- results remain directionally stable in full-text-only and paper-blocked
  analyses; and
- model performance meaningfully exceeds mean, heuristic, and
  publication-volume baselines.

Until these criteria are met, improvements should be described as exploratory
retrieval gains rather than validated biological similarity prediction.
