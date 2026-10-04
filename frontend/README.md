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

## API

`api/server.py` uses only the standard library, plus v2's own dependencies
for the model.

| Endpoint | |
|---|---|
| `GET /api/health` | `{"status": "loading" \| "ready" \| "error", "diseases": n}` |
| `POST /api/place` | `{"disease": {...}, "id": "USER:...", "top": 20, "others": [{"id", "disease"}]}` → neighbours with v2's explanation, support, warnings and refitted weights |
| `GET /api/suggest?field=phenotypes\|genes\|drugs\|ontology&q=...` | vocabulary matches for the form |

```sh
npm run test:api   # request handling, with a stub in place of the model
```
