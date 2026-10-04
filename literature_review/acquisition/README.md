# Scientific literature acquisition

This collector implements the **acquisition portion of phase 1** in
`experiments/SCIENTIFIC_LITERATURE_ACQUISITION_METHOD.md`. It creates an actual
SQLite database and a licensed full-text corpus, not synthetic paper records.
It does **not** turn co-mentions or the older CSV's similarity values into
supervised labels.

## Resumable relationship acquisition

`acquire_relationships.py` is the larger dual-source collector. It searches
Europe PMC and PubMed E-utilities for up to 5,000 real records, deduplicates
them by PMID, PMCID, DOI, then normalized title, and maps exact disease names to
the local Orphanet release. Ambiguous aliases remain unmapped and enter the
review queue.

The pipeline first detects disease mentions and short candidate passages. It
then assigns independent automated scores for clinical phenotype, genetic
etiology, molecular mechanism, natural history, and therapeutic similarity.
Missing dimensions remain `NA`. Abstract-derived scores always have confidence
1, and every automated score or `NA` remains `unverified`.

```powershell
python literature_review/acquisition/acquire_relationships.py run --max-papers 5000
python literature_review/acquisition/acquire_relationships.py validate
python literature_review/acquisition/acquire_relationships.py report
python -m unittest literature_review/acquisition/test_acquire_relationships.py -v
```

The default database is `.data/literature_acquisition/papers.sqlite`; raw API
responses and `acquisition_report.json` are stored beside it. Successful
responses are checksum-verified and reused on subsequent runs, while failed
requests are retried. PubMed traffic is limited to three requests per second
without an API key and ten with `NCBI_API_KEY`; Europe PMC requests are also
serialized and rate-limited. Full text is requested only from the Europe PMC
open-access XML endpoint for records marked open access.

## Acquired release: 2026-10-04

The completed pilot contains 1,973 distinct paper records from 370 successful
search/citation log entries. It downloaded 180 open-access XML articles; 107
passed the conservative reusable-text license filter. The text export contains
4,967 paragraphs (545,911 whitespace-delimited words). The review queue contains
166 candidate passages covering 43 disease pairs, including 18 of the 30
registered pairs. There are 54 canonical diseases and 55 pairs in the database
when registered pairs without extracted passages are included.

The provisional text split has 105 training papers / 4,893 paragraphs and two
test papers / 74 paragraphs, with **no validation papers**. This imbalance is
retained rather than breaking connected components. It is not an adequate
benchmark. There are **zero reviewed similarity scores or verified controls**.
One provider-flagged retracted record was retained for audit and excluded from
training. Ten historical failed requests remain logged; all final queries
succeeded. Of the successful log entries, 347 stopped at registered pilot page
caps, so this is not an exhaustive search.

SQLite integrity, all cached-response checksums, component split integrity,
20 sampled source-locator/attribution comparisons, and seven focused unit tests
passed. No identical exported paragraph appears in multiple splits. The local
release occupies approximately 114 MB, including its 27 MB SQLite database.
`qa_report.json` records these checks and the database checksum.

`registered_protocol.json` beside this guide is a copy of the preregistration;
the authoritative frozen copy and original responses are in the release directory.

## Local outputs

The default release directory is `.data/literature_acquisition/v1/`:

- `literature.sqlite3`: papers, canonical diseases and identifiers, discovery
  routes, search logs, raw-response provenance, licensed body paragraphs,
  candidate pair passages, review tables, and provisional split assignments.
- `protocol.json`, `protocol.sha256`, `rubric.md`: registration and frozen rubric.
- `raw/*.response`: original responses, saved before parsing, with checksums.
  Responses include failed requests. This is a local provenance cache; do not
  redistribute abstracts or license-restricted articles as training text.
- `exports/text_training.jsonl`: CC-BY/CC0 body paragraphs with title, authors,
  journal, year, DOI/PMCID, license, source locator, checksum and attribution.
  Intended for exploratory domain adaptation or retrieval. One paragraph per
  row, with provisional paper/pair/family component assignments.
- `exports/review_queue.jsonl`: unscored, source-located candidate passages.
  Co-mention identifies material for review, not substantive evidence. Hinted
  dimensions are search aids; reviewers must assess all seven dimensions.
- `exports/screening_queue.csv`: paper-level screening worksheet. Each reviewer
  should use an independent copy; blank columns are not review votes.
- `exports/pair_search_coverage.csv`: executed queries, result caps and retrieved
  passages per registered pair.
- `exports/supervised_training.jsonl`: intentionally empty until calibration,
  independent review, adjudication and release gates have been implemented and met.
- `release_manifest.json`: counts, coverage, failures, checksums and limitations.

These data files are under the project's existing `.data/` ignore rule. Back up
the release directory along with the code if the acquired corpus must be retained.

## Reproduce or resume

Python 3.10+ and the standard library are sufficient. From the project root:

```powershell
python literature_review/acquisition/acquire.py register
python literature_review/acquisition/acquire.py run
python literature_review/acquisition/acquire.py validate
python -m unittest discover -s literature_review/acquisition -p test_acquire.py -v
```

Run `register` only once per output directory. `run` resumes cached successes and
retries failed queries. `export` rebuilds candidates' existing splits and exports
without network calls. Use `--output <new-directory>` for a new registration.
Do not run simultaneous collectors against the same release. The collector
verifies the registration and rubric hashes and refuses silent edits to either.

The registered search range is 2000-01-01 through 2026-10-04. The Orphanet snapshot
is identified by its actual XML date, version and checksum. Selection takes 30
pairs from exact unambiguous names in the pre-existing CSV, requires compatible
entity types, and limits disease degree to two. Scores and findings from that
CSV are never imported. This sampling remains biased toward previously studied
pairs and is not a representative universe sample. Orphanet membership alone
does not verify prevalence below 1/2,000 in every population.

The collector executes all registered A, B and C query templates against the
[Europe PMC REST API](https://europepmc.org/RestfulWebService), with one request
at a time, retries and local caching. It uses the API's first relevance page;
queries capped at 20 records are explicitly marked incomplete. It explores two
bounded citation rounds from provisional OA review candidates and attempts up
to 180 OA XML downloads. These are acquisition budgets, **not evidence saturation**.
Independent PubMed, Crossref and Unpaywall queries and related-article expansion
are deferred. Europe PMC already supplies PubMed metadata, but that does not
constitute an independent PubMed search.

## Review and release requirements

1. Have two independent, domain-informed reviewers screen at least 100 papers
   (or 10% of the batch, whichever is larger). Preserve votes in
   `paper_screening_votes` and full-text candidate votes in `screening_votes`.
   Resolve mapping/subtype ambiguity using the original article, not string
   matching alone. Apply the protocol's controlled exclusion reasons.
2. Independently extract atomic dimension observations for the pilot, including
   population, model system, study design, sample sizes, quality components,
   ordinal score, applicability, confidence, exact source locator and reviewer.
   Preserve missing scores as NULL. Read surrounding text, tables, supplements,
   negative results and cited primary studies; a paragraph may omit qualifiers.
3. Confirm corrections/retractions and underlying study families. Every initial
   `UNRESOLVED:*` family is a placeholder, **not independent replication**. Merge
   cohorts, preprints and primary/secondary reports before final splitting.
4. Measure screening kappa and ordinal agreement. The registered screening
   threshold is 0.70; within-one-level score agreement target is 0.90. Training
   cannot relax to one extractor plus audit before calibration is demonstrated.
5. Preserve both extractions and adjudicate conflicts independently. Construct
   verified unrelated and related-but-distinct controls from explicit evidence.
   Current pairs are `unknown`; no missing edge becomes an unrelated control.
6. Implement the registered family-capped weighted-median consensus and audited
   supervised exporter after review. The schema supports these records, but
   this acquisition release does not claim completed aggregation or model training.
7. Freeze a reviewed benchmark and recompute splits. The collector's mechanical
   paper/pair/family split uses connected components, including direct-query
   pair links and exact shared long paragraphs. Giant components are allowed;
   do not split them to force a cosmetically balanced train/test ratio.
   Disease-held-out assignments quarantine cross-boundary papers. Time splits
   use <=2022 / 2023 / >=2024, quarantining families crossing date boundaries.
   All assignments remain provisional until family/mapping review. Do not fit
   preprocessing or models on validation/test material.
8. Do not report primary biological-similarity performance without the protocol's
   minimum held-out evidence, control balance and baseline/sensitivity analyses.

The raw corpus and review queue can be used now. **This is not a release of
validated disease-similarity ground truth.** No automated process can truthfully
substitute fabricated reviewers for the protocol's independent calibration.

## Licensing

Europe PMC permits full-text acquisition through its documented API for its
[open-access subset](https://europepmc.org/downloads/openaccess). Licenses vary
by article. The reusable-text export conservatively accepts an explicit CC-BY
or CC0 URL in the article's license statement and excludes conflicting NC, ND
or SA terms. Public availability or an `isOpenAccess` flag alone is insufficient.
Retain attribution and license fields when using the export. This filter does
not infer downstream rights for abstracts, figures, supplementary datasets or
third-party material. Bodies are supplied as normalized text; figures and
reference lists are not exported.
