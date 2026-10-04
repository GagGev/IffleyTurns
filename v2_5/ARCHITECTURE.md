# V2.5: evidence-grounded paper extraction with MedGemma

## Scope

V2.5 adds a hybrid paper-to-profile layer in front of the existing v2
similarity model. It does not retrain or replace v2's graph encoders or fusion
model.

```
paper
  -> passage selection (local, deterministic)
  -> deterministic exact extraction
  -> MedGemma 27B structured extraction
  -> evidence and vocabulary validation
  -> merged v2 disease record
  -> existing v2 encoder, fusion, ranking, and explanations
```

This separation makes it possible to measure whether MedGemma improves the
paper-input benchmark without changing the graph model being evaluated.

## Trust boundary

- The default endpoint must be on localhost. Sending unpublished papers to a
  remote endpoint requires an explicit `--allow-remote` flag.
- Complete papers are not sent by default. A deterministic selector retains
  the title/abstract and high-value phenotype, gene, mechanism, natural-history
  and treatment passages within a configurable word budget.
- Every accepted LLM feature must cite a passage locator and a verbatim quote
  found in that passage.
- IDs are accepted only if they exist in the loaded v2 knowledge base.
- Invalid, ambiguous, unsupported, negated, or comparator-only features are
  rejected and retained in an audit list.
- LLM output is always labelled `automated` and `unverified`.
- Failed LLM calls fall back to the deterministic v2 paper extractor.

## Extracted schema

MedGemma returns JSON containing:

- focal-disease label, if explicitly stated;
- phenotypes with HPO IDs;
- genes and their asserted relationship;
- drugs and treatment role;
- inheritance;
- age of onset;
- prevalence per person;
- one short disease description;
- an evidence locator, verbatim quote, and confidence from 1–3 for each item.

The LLM cannot add ontology parents or Open Targets association profiles.
Pathway and drug-target modalities continue to be derived by v2 from accepted
genes and drugs.

## Grounding policy

- HPO IDs: canonicalized and checked as phenotypic-abnormality terms.
- Genes: exact, case-sensitive membership in v2's gene-symbol catalogue.
- Drugs: resolved to a unique parent ChEMBL identifier.
- Inheritance and onset: restricted to v2's controlled values.
- Prevalence: finite fraction in `(0, 1]`.
- Evidence: locator must exist and normalized quote must occur in the supplied
  passage.

Deterministic matches are supplied to MedGemma as canonical-ID candidates, not
as established facts. On a successful LLM call, only evidence-grounded features
affirmed for the focal disease enter the record; this prevents a gene or drug
mentioned only for a comparator from leaking through the exact matcher. The
deterministic structured features are used directly only when MedGemma is
disabled or fallback is explicitly allowed. The masked source text, not an LLM
summary, remains the v2 text modality.

## Inference backends

V2.5 supports two interchangeable JSON-completion backends.

### Direct Transformers

`--backend transformers` loads `google/medgemma-1.5-4b-it` directly with
Hugging Face Transformers. The default is NF4 4-bit quantization with double
quantization and automatic BF16/FP16 compute selection. PyTorch, Transformers,
bitsandbytes and CUDA are imported lazily on the first model request: importing
v2.5, displaying command help, and running CPU tests cannot initialize the GPU
or download weights.

### OpenAI-compatible endpoint

`--backend endpoint` allows the 27B model to be served by vLLM, SGLang,
llama.cpp, LM Studio, or another runtime without importing its GPU stack into
the graph process. Endpoint defaults:

- base URL: `http://127.0.0.1:8000/v1`
- model: `google/medgemma-27b-text-it`
- response mode: strict JSON Schema
- temperature: `0`
- response cache: `.data/v2_5/medgemma_cache`

Both backends expose the same interface. The selected backend, model,
quantization, immutable model revision when supplied, and response hash are
stored with benchmark outputs.

## Evaluation

`benchmark_medgemma.py` reuses the frozen v2 paper-input cohort and strict
joint holdout. It masks recognized disease names, caches each disease/level
extraction independently, and can resume after interruption.

Primary comparison:

- deterministic masked extraction from v2;
- v2.5 hybrid MedGemma extraction;
- hidden-name database profile oracle.

Metrics remain MAP, MRR, Hits@10, Recall@10, Recall@50 and nDCG@10 against
curated neighbours. Same-paper comparators remain secondary silver labels.

