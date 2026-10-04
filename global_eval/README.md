# global_eval

One place to compare every model version (v1, v2, v3) on benchmarks none of them owns. It imports no version
directly: each is scored in its own process (their module names collide), and the benchmarks only read the score
matrices those processes write.

```bash
python global_eval/run.py                 # score (cached) and evaluate everything -> global_eval/results/
python global_eval/run.py --rescore       # recompute the score matrices (about 5 minutes on a laptop CPU)
python global_eval/run.py --only paper_pairs relations forward_time
python -m pytest global_eval/tests -q
```

Needs the v2 and v3 environments (pandas, scikit-learn, scipy, PyTorch for v3) and the built v2 and v3 models
(`v2/.data/model`, `v3/.data/model` plus `v3/.data/cache`). Score matrices go to `.data/global_eval/` (large,
untracked); reports and metrics go to `global_eval/results/` (tracked).

## Layout

| Path | What |
|---|---|
| `run.py` | Orchestrator: scores, evaluates, writes `results/report.md` and `results/metrics.json` |
| `core.py` | Model registry, retrieval metrics (MAP, MRR, Hits@10, P@10, nDCG@10, AUROC), bootstrap CIs |
| `scorers/v2_side.py` | Defines the tasks; scores v1 and v2 (imports `v2/`) |
| `scorers/v3_side.py` | Scores v3 production static similarity and forecast (imports `v3/`) |
| `scorers/paper_pairs_data.py` | Loads the paper-stated pairs from `literature_review/` |
| `benchmarks/paper_pairs.py` | Benchmark 1 |
| `benchmarks/relations.py` | Benchmark 2 |
| `benchmarks/symptom_retrieval.py` | Benchmark 4: symptom-only queries (the patient-view scenario) |
| `benchmarks/forward_time.py` | Benchmark 3 (v3's temporal protocol) |
| `review_sheet.py` | Blinded expert-review sheet and key (`results/expert_review/`) |
| `results/` | `report.md`, `metrics.json`, `paper_pairs_per_pair.json`, v3 forward-time outputs |

## Benchmarks

1. **Paper-stated pairs.** About 560 disease pairs that papers describe as similar or related (both literature
   datasets merged, ambiguous ORPHA mappings dropped). Each model ranks the 7,493 catalogue diseases for each member
   of a pair; we report where the partner lands. No model was trained on these labels, except for the 58% of pairs
   that also have a curated relation (reported as a separate stratum, with the 42% that do not).
2. **Curated relations.** Orphanet siblings, shared causal gene, shared trial drug; 400 sampled query diseases each,
   with the modalities that define the relation hidden (as in `v2/benchmarks.py`). **In-sample for v2 and v3**
   (their production models were fitted on these relations), so read the gap to v1 as fit, not generalisation.
3. **Forward in time.** v3's protocol: models trained on knowledge before a cutoff predict FDA/EMA orphan
   designation relations formed afterwards. It needs models retrained at each cutoff, so it is produced by
   `v3/run_evaluation.py` and imported here, not recomputed from the production weights (trained through 2026).

4. **Symptom-only retrieval.** 3, 5 or 10 of a disease's own phenotypes (or 5 plus one unrelated term) and nothing
   else, for 500 diseases. Does the disease, or one of its Orphanet siblings, come back in the top 10? This is the
   patient view's situation and the cold-start case for papers that yield few features.

## Not covered yet

- **Papers as input (v2_5).** The paper-input benchmark in `v2/PAPER_INPUT_BENCHMARK.md` needs acquired full texts and
  MedGemma; nothing here tests the upload path end to end.
- **Expert judgement.** Ground truth lists only relations somebody recorded. `review_sheet.py` produces a blinded sheet
  of 377 neighbour pairs for clinicians to rate; the ratings are not in yet.
- **Standard external baselines** (Resnik/Phenomizer-style HPO similarity) and a popularity prior for paper pairs.
- **Seed variance of v3 training** (the forward-in-time numbers moved by about 0.005 MAP between the README's run and
  this one).
- **Annotation date leakage** (HPO/ClinGen dates) for the static modalities.

## Models

| Id | What |
|---|---|
| `v1` | Weighted Jaccard over raw annotation sets plus prevalence |
| `v1_drugfree` | The same without approved drugs |
| `v2` | v2 production fusion over all 12 modalities |
| `v2_drugfree` | The same with drug, drug-target and Open Targets modalities hidden (the view v3 works in) |
| `v3_static` | v3 production neural ensemble + fusion (drug-free); the graph's similarity score |
| `v3_forecast` | v3 production stacker: the score that a pair will be linked by a future orphan designation |

## Adding a model or benchmark

Add the model to `core.MODELS`, have a scorer write `scores/<task>/<id>.npy` (queries in the order of
`tasks.json`, columns in catalogue order), and list it in the benchmark's `MODEL_ORDER`. A new benchmark is a module
in `benchmarks/` with `NAME`, `TITLE`, `INTRO` and `run(tasks) -> (metrics, markdown)`, registered in `run.py`.
