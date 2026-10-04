# Rare Disease Relationship Explorer

A React frontend for researchers to explore relationships between rare diseases.

It works today without the Python backend. The graph is built from the
literature-review CSVs already in the repository. Feature-based similarity from
`evaluation.py` is shown as "not connected" until a backend exists.

## Run

```sh
cd frontend
npm install
npm run dev        # rebuilds the data, then serves on http://localhost:5173
```

`npm run dev` and `npm run build` first run `scripts/build_literature_graph.py`
(Python 3, standard library only), which writes `public/data/literature_graph.json`.
Run `npm run data` by hand after editing the literature CSVs.

## What it shows

- **Graph**: diseases as nodes and literature-backed pairs as edges. Edge colour
  is the dominant relationship (similar, related but distinct, unrelated), width
  is the number of papers, and strength of colour is the score.
- **Table**: the same pairs, sortable and keyboard accessible.
- **Disease panel**: ORPHA link, aliases, and related diseases ranked by score.
- **Pair panel**: overall similarity, evidence strength (0–15), evidence caveats,
  consensus score for each of the 7 similarity dimensions, and every paper with
  its finding and PubMed link.
- **Filters**: score by overall or a single dimension, minimum score, evidence
  strength and paper count, relationship type, rarity, and evidence source.
- **Neighbourhood** mode limits the view to 1–3 steps around the selection.
- **Export CSV** downloads the pairs currently in view.
- The URL hash records the selection, so a view can be bookmarked or shared.
- **Add a disease**: a form or a JSON upload places a researcher's own disease in
  the graph. It links (dotted aqua lines, diamond node) to its most similar
  diseases by features, and its panel ranks every comparable disease with the
  shared terms behind each score. Added diseases are kept in the browser's local
  storage. **Export JSON** saves them, and the file can be uploaded again later.

## Data

`scripts/build_literature_graph.py` merges two sources:

| Source | Files |
|---|---|
| Literature review | `literature_review/literature_disease_pairs.csv`, `paper_dimension_scores.csv` |
| Pair-first run | `literature_review/additional_runs/*_pairfirst.csv` |

Diseases are merged by ORPHA ID. Rows with a missing or ambiguous ORPHA mapping
become name-keyed nodes, and the disease panel flags them. Edge aggregates use
the same scoring as `literature_review/build_edge_ranking.py`. Dimension
consensus is the paper-weighted mean used by `models/linear_regression_7_dimension.py`.
Aggregates are precomputed for each evidence-source selection, so the frontend
never re-implements the scoring.

## Feature-based similarity

Similarity between feature profiles is computed in the browser by
`src/lib/similarity.ts`, a port of the metric in `evaluation.py`. It uses the
same feature families, weights, Jaccard overlap, prevalence decay and
renormalisation over the features both diseases have. Check that the two still
agree after changing either one:

```sh
npm run check:similarity   # compares both on 1,830 random pairs
```

Feature profiles for catalogue diseases come from the backend's feature tables:

```sh
python download_databases.py && python generate_features.py   # from the repository root
cd frontend && npm run features   # writes public/data/disease_features.json (needs pyarrow)
```

Until that file exists, the header shows "Feature profiles: not generated".
Added diseases can then only be compared with each other. Once it exists:
- added diseases are compared with every Orphanet disease;
- each disease panel lists its feature-based nearest neighbours;
- each pair panel shows the feature breakdown next to the literature evidence;
- form fields suggest HPO terms, body systems, categories and drugs by name.

### Uploading diseases as JSON

Upload one object, a list, or `{"diseases": [...]}`. Field names match
`diseases.parquet`, so backend rows can be uploaded as they are. Only `name` is
required:

```json
{
  "name": "Example disease",
  "orpha_id": "ORPHA:558",
  "status": "rare",
  "hpo_ids": ["HP:0001166"],
  "gene_symbols": ["FBN1"],
  "inheritance": ["Autosomal dominant"],
  "onset": ["Childhood"],
  "category_ids": [],
  "body_system_ids": [],
  "approved_drug_ids": [],
  "prevalence_class": "1-5 / 10 000"
}
```

`prevalence_estimated_per_person` (a number between 0 and 1) can be given
instead of `prevalence_class`. Invalid values are dropped and unknown fields
ignored, and the upload preview lists both.

## Connecting the backend

All data access goes through `src/data/source.ts`. A backend only needs to
serve the same two documents: the literature graph and the feature catalogue.
Comparisons happen in the browser, so researchers can add diseases without
sending them to a server.
