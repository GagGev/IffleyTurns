# V2.5

V2.5 uses MedGemma to turn a paper into an evidence-grounded disease profile,
then sends that profile through the existing v2 similarity model. It supports
the official 4B model directly in Python and any MedGemma model exposed through
an OpenAI-compatible endpoint. V2's trained graph model, catalogue features,
and literature database are not modified.

See `ARCHITECTURE.md` for the extraction and trust model.

## 1. Direct Python backend (recommended for an 8 GB GPU)

The direct backend loads the official lightweight model in 4-bit mode:

```text
google/medgemma-1.5-4b-it
```

Access to the official weights may require accepting Google's model terms and
authenticating with Hugging Face. Accept the terms on the model page, install a
CUDA-enabled PyTorch build using the official PyTorch selector, then install:

```powershell
python -m pip install --upgrade transformers accelerate bitsandbytes huggingface_hub
```

Configure authentication and the backend:

```powershell
$env:HF_TOKEN = "<your Hugging Face token>"
$env:MEDGEMMA_BACKEND = "transformers"
$env:MEDGEMMA_MODEL = "google/medgemma-1.5-4b-it"
```

Run a placement:

```powershell
python -m v2_5.place_paper `
  --backend transformers `
  --quantization 4bit `
  --paper-id 10096 `
  --strict-llm `
  --top 10
```

Imports, command help, and deterministic tests do not load the model. The
first command that actually requests a MedGemma extraction downloads the
weights, initializes CUDA, and caches the model through Hugging Face.

Use `--local-files-only` after the weights are cached to prevent network
access. The direct backend uses greedy decoding, validates the JSON after
generation, and rejects every unsupported feature.

## 2. OpenAI-compatible endpoint backend

The endpoint backend remains available for the 27B model or a shared GPU
server. For example, on a supported Linux/WSL2 vLLM installation:

```powershell
$env:HF_TOKEN = "<your Hugging Face token>"
vllm serve google/medgemma-27b-text-it `
  --host 127.0.0.1 `
  --port 8000 `
  --dtype bfloat16 `
  --max-model-len 8192
```

The unquantized 27B checkpoint generally needs a high-memory accelerator.
Quantized deployments can fit smaller GPUs, but the quantized checkpoint,
runtime, model terms, and extraction quality must be reviewed separately.

Configure another compatible runtime with:

```powershell
$env:MEDGEMMA_BASE_URL = "http://127.0.0.1:8000/v1"
$env:MEDGEMMA_MODEL = "google/medgemma-27b-text-it"
$env:MEDGEMMA_MODEL_REVISION = "<immutable model revision>"
$env:MEDGEMMA_API_KEY = ""
```

JSON Schema mode is the default. If the server does not implement it, try
`--response-mode json_object`; `prompt_only` is the least reliable fallback.

V2.5 refuses non-local endpoints unless `--allow-remote` is explicitly given.
Do not send confidential or unpublished manuscripts to a remote service
without an appropriate data agreement.

## 3. Place a paper in the v2 graph

Text, JSON, JATS/XML, optional PDF, and acquired-corpus inputs are supported.

```powershell
python -m v2_5.place_paper `
  --backend transformers `
  --input manuscript.txt `
  --disease-name "Example disease" `
  --top 20 `
  --output .data/v2_5/example-placement.json
```

```powershell
python -m v2_5.place_paper `
  --backend transformers `
  --paper-id 123 `
  --top 20 `
  --output .data/v2_5/paper-123-placement.json
```

PDF parsing is optional:

```powershell
pip install pypdf
```

By default, the supplied disease name is only a display label and does not
affect similarity. `--use-name-modality` enables name-based matching.

If MedGemma is unavailable, interactive placement falls back to the
deterministic v2 extractor and reports that status. Use `--strict-llm` to fail
instead. `--deterministic-only` is useful for an offline smoke test.

## 4. Evaluate v2.5

Start with a small, single-budget run:

```powershell
python -m v2_5.benchmark_medgemma `
  --backend transformers `
  --quantization 4bit `
  --levels first_2000_words `
  --max-queries 10
```

Then run the frozen 200-paper benchmark:

```powershell
python -m v2_5.benchmark_medgemma `
  --backend transformers `
  --quantization 4bit `
  --levels title,title_abstract,first_500_words,first_2000_words,full_text
```

Each extraction is saved under:

```text
.data/v2_5/paper_input_benchmark/extractions/
```

Both extraction files and model responses are cached. Re-running the command
resumes completed work. Use `--refresh` only when intentionally changing the
prompt, model revision, or backend configuration.

Benchmark mode is strict by default: model failures stop the run instead of
silently scoring deterministic records. `--allow-fallback` is available for
diagnostics, but fallback rows must not be reported as MedGemma results.

To score already completed extractions without model calls:

```powershell
python -m v2_5.benchmark_medgemma --score-only
```

Outputs include:

- `REPORT.md`: comparison with the deterministic v2 paper extractor;
- `summary.json`: aggregate metrics and metadata;
- `per_query.csv`: query-level ranking metrics and extraction status;
- `progress.json`: resumable progress;
- `extractions/`: evidence, rejected features, model hash and provenance.

## Extraction guarantees

- no complete-paper upload by default: selected passages are capped at 2,000
  words;
- evidence locator and verbatim quote required for every accepted LLM feature;
- HPO, gene, Reactome and ChEMBL identifiers validated against v2 knowledge;
- comparator-only, negated, ambiguous and unsupported features rejected;
- accepted rows marked `automated` and `unverified`;
- response cache and extraction files written atomically;
- v2 ranking explanations remain available for every returned neighbour.

This remains a research pipeline, not a diagnostic or clinical decision
system. Human review is required before adding extracted evidence to a graph
used for scientific conclusions.

