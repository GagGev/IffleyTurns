# Rare Disease Similarity Explorer

A React frontend that lets researchers explore the **v2** rare-disease
similarity graph and place new diseases in it.

Everything shown comes from the v2 model in `../v2`. That covers the edges,
scores, percentiles, support levels and explanations, and the placement of new
diseases. The frontend does not compute similarity itself.

## Setup

1. Build the v2 graph (see `v2/README.md`). From the repository root:

   ```sh
   python download_databases.py && python generate_features.py
   python v2/run_evaluation.py
   python v2/build_graph.py
   ```

2. Convert it for the frontend:

   ```sh
   cd frontend
   npm install
   npm run data     # v2/.data/graph -> public/data (graph.json + detail shards)
   ```

   Then write the per-disease annotations and the literature pairs, which the
   gene, symptom, onset and inheritance colouring and the paper checks use. This
   needs the v2 Python environment (it reads v2's cached knowledge bundle):

   ```sh
   npm run data:annotations   # -> public/data/annotations.json + literature.json
   ```

   If pyarrow and scikit-learn are installed, `npm run data` also computes a
   t-SNE layout from v2's fused embedding. That takes about a minute, and the
   browser then shows the 7,500-disease graph without simulating it. Without
   them, the browser lays the graph out itself, which is slower.

3. Start the placement service, using the Python environment you used for v2:

   ```sh
   python frontend/api/server.py   # from the repository root; http://127.0.0.1:8765
   ```

4. Run the app:

   ```sh
   npm run dev      # proxies /api to the placement service
   ```

   Or run `npm run build` once. `frontend/api/server.py` then serves the built
   app as well, at http://127.0.0.1:8765, which is the simplest option for
   researchers.

Until step 1 has run, the app shows these instructions. Without the placement
service, the graph still works, and added diseases are saved but not placed
until the service is running.

## What it shows

- **Graph**: every Orphanet disease in v2's cohort linked to its 10 most
  similar diseases. Edge colour is v2's support level:
  - **curated**: an Orphanet relation, shared causal gene or shared trial drug backs the edge;
  - **plausible**: the diseases share an Orphanet group, a gene or a drug;
  - **novel**: none of these, so the edge is a hypothesis to review.

  By default only mutual edges are drawn, where both diseases list each other.
- **Filters**: support level, mutual only, minimum percentile against random
  pairs, the modality that contributes most (for example drugs, for repurposing
  leads), and Orphanet category.
- **Disease panel**: which of the 12 modalities are annotated, and the most
  similar diseases.
- **Pair panel**: the percentile, score and support level, any curated
  relations, the shared features behind the top modalities (phenotypes, genes,
  pathways, drugs and so on), and each modality's contribution to the score.
- **Node colour**: Louvain cluster (default), Orphanet category, one gene, one
  symptom (an HPO term and its subtypes, darker where more patients have it),
  age of onset, mode of inheritance, or similarity to a chosen disease (by ORPHA
  ID or name). Gene, symptom, onset and inheritance modes dim every disease that
  does not match. Hover or select a disease for its ORPHA ID and cluster.
- **Papers**: upload a paper (PDF, text, XML) or a v2_5 result file
  (`python -m v2_5.place_paper --output result.json`). A paper's focal disease is
  placed in the graph, and its claims are drawn over the graph and checked (see
  below). Uploading a PDF or text file currently runs a **mock** extractor
  (`src/lib/paperService.ts`), which builds a v2_5-shaped result from the graph;
  swap `mockExtractPaper` for a call to the real pipeline. The **Curated
  literature** panel lists 508 papers from `literature_review/additional_runs` with 539
  paper-stated disease pairs.
- **Find a paper** searches, by title or PMID, the uploads, the curated literature,
  and the 1,972 other ("newly acquired") papers of the latest acquisition release
  (`.data/literature_acquisition/v1`, 2026-10-04): 2,480 papers in all. The
  release is unreviewed, so for its papers only the 86 disease pairs that open
  full text co-mentions are shown, and only whether the graph has the edge is
  checked.
- **Does the graph carry the paper's evidence?** Each claim says two diseases
  are similar in some respects (phenotype, genes, pathways, treatment,
  epidemiology). It is judged against the graph's edge for that pair: *no edge*
  if neither disease lists the other among its top 10; otherwise *carried* if
  every named dimension gets at least 0.25 logit from one of v2's matching
  modalities, *partly carried* if some do, *edge, other reasons* if none do, and
  *edge present* when the dimension (diagnostic, comorbidity) has no v2 modality.
  Verdict colours are drawn on the edges; dashed grey lines are claims with no
  edge. For an uploaded paper, each extracted gene, phenotype, onset and
  inheritance value is also checked against the paper disease's closest
  neighbours. The literature pairs were selected partly by phenotype similarity,
  so the recovery rate there is optimistic.
- **Table view**, **CSV export**, **neighbourhood mode**, and shareable links:
  the URL hash records the selection.

## Adding a disease

**Add a disease** takes a form or a JSON upload. The JSON is v2's own input
format, the same one `python v2/place_disease.py --json` accepts:

```json
{
  "name": "...", "description": "free text", "synonyms": ["..."],
  "phenotypes": {"HP:0001250": 1.0, "HP:0001263": 0.8},
  "genes": ["CDKL5"], "drugs": ["CHEMBL1234 or a drug name"],
  "inheritance": ["X-linked dominant"], "onset": ["Infancy"],
  "prevalence": 1e-6, "ontology_parents": ["ORPHA:102369"]
}
```

Only `name` is required. Uploads may also include v2's optional `pathways`
(Reactome IDs). The form autocompletes HPO terms, genes, drugs and
classification parents from v2's vocabulary, and asks how often each phenotype
occurs, which v2 uses as the phenotype weight. "Preview matches" shows the
closest diseases before saving.

The placement service runs v2's `record_from_user_input` and `place`. v2
refits its fusion to the modalities the new disease has, so the scores match
`place_disease.py`. Warnings about ignored values are shown on the disease.
Each added disease is also matched against the user's other added diseases.

Added diseases are kept in the browser's local storage and are never written
to the v2 graph files. **Export JSON** saves them, and **Download v2 JSON**
on a disease gives a file for `place_disease.py --json --add` to add it to the
shared graph.

## Patient view

The home page offers two views. The **researcher view** is the explorer
described above. The **patient view** is for patients and families. It is
deliberately simple and framed throughout as "not a diagnosis":

1. Describe symptoms in your own words. MedGemma turns them into candidate
   clinical terms, each is matched to an HPO term, and the patient confirms or
   removes them. Symptoms can also be chosen from a list.
2. The confirmed symptoms are placed with the v2 model, the same way as **Add a
   disease**. The closest diseases are grouped by cluster, and the top three
   groups are shown.
3. Each group shows a plain-language name, a short explanation from MedGemma,
   example conditions, and **who to talk to**. Each example shows the symptoms
   it shares, an Orphanet link and an "Explain simply" button. That button
   rewrites Orphanet's own description; it is not the model's own knowledge.
   Each example also lists up to two **key papers**: the most-cited Europe PMC
   papers with the condition's name, or a specific synonym, in the title. Only
   the public condition name is sent to Europe PMC. Abbreviations, broad
   synonyms and animal studies are skipped. When nothing reliable is found, a
   Europe PMC search link is shown instead.
   General resources (GP, Orphanet, Genetic Alliance UK, EURORDIS, NORD) follow.

MedGemma only handles language. Which conditions match is decided by v2. Group
names and the "who to talk to" advice come from a fixed table in
`src/data/patient.ts`, not from the model. Safeguards on MedGemma's terms:

- each term is checked against the HPO vocabulary, and unmatched terms are left
  for the patient to search;
- a term is never made more specific than the patient said;
- a match that negates the term ("Lack of skin elasticity" for stretchy skin) is
  rejected;
- the patient confirms every term before it is used.

### Running MedGemma

The patient view uses a text-only MedGemma checkpoint through any
OpenAI-compatible endpoint on this machine. The included server loads it with
transformers (Apple GPU, CUDA or CPU):

```sh
pip install torch transformers   # if not already installed
python frontend/api/medgemma_server.py --model /path/to/medgemma-4b-text   # http://127.0.0.1:8766/v1
```

On an M4 MacBook the 4B model loads in about 10 seconds, uses about 8 GB of
memory, and answers in 3–10 seconds. llama-server or vLLM can be used instead;
set `MEDGEMMA_BASE_URL` (it must be a local address, so symptoms never leave
the machine). Without MedGemma, the patient view still works: symptoms are
chosen from a list and the fixed group descriptions are shown.

## API

`api/server.py` uses only the standard library, plus v2's own dependencies
for the model.

| Endpoint | |
|---|---|
| `GET /api/health` | `{"status": "loading" \| "ready" \| "error", "diseases": n}` |
| `POST /api/place` | `{"disease": {...}, "id": "USER:...", "top": 20, "others": [{"id", "disease"}]}` → neighbours with v2's explanation, support, warnings and refitted weights |
| `GET /api/suggest?field=phenotypes\|genes\|drugs\|ontology&q=...` | vocabulary matches for the form |
| `GET /api/patient/status` | whether MedGemma is reachable |
| `POST /api/patient/interpret` | `{"text"}` → clinical terms with HPO matches for the patient to confirm |
| `POST /api/patient/explain-disease` | `{"id", "name"}` → Orphanet's description in plain words (`text` is null if there is none) |
| `POST /api/patient/explain-group` | `{"category", "examples", "shared"}` → a group in plain words |
| `GET /api/patient/papers?id=ORPHA:x` | up to two most-cited papers about a disease, plus a Europe PMC search link |

```sh
npm run test:api   # request handling and patient helpers, with stubs in place of v2 and MedGemma
```
