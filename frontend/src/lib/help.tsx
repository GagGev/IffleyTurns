// Definitions behind the "?" buttons in the sidebar.  Kept in one place so the
// wording can be reviewed and edited without touching the components.

import type { ReactNode } from 'react'
import { VERDICT_LABEL, VERDICT_MEANING, VERDICT_ORDER } from './paperEval'

export type HelpTopic =
  | 'search'
  | 'paperSearch'
  | 'yourDiseases'
  | 'linksPerDisease'
  | 'papers'
  | 'literatureSet'
  | 'filters'
  | 'support'
  | 'mutual'
  | 'score'
  | 'mainEvidence'
  | 'cluster'
  | 'category'
  | 'symptoms'
  | 'nodeColour'
  | 'readingGraph'
  | 'diseasePapers'

const verdictList = (
  <dl className="help-list">
    {VERDICT_ORDER.map((v) => (
      <div key={v}>
        <dt>{VERDICT_LABEL[v]}</dt>
        <dd>{VERDICT_MEANING[v]}</dd>
      </div>
    ))}
  </dl>
)

export const HELP: Record<HelpTopic, { title: string; body: ReactNode }> = {
  search: {
    title: 'Find a disease',
    body: (
      <p>
        Search the diseases in the graph by name or ORPHA ID.
        Choosing one zooms the graph to it and its most similar diseases, and opens its details.
      </p>
    ),
  },
  paperSearch: {
    title: 'Find a paper',
    body: (
      <>
        <p>
          Search papers by title or PMID. Choosing one opens it and draws the disease pairs it links on the graph, in the
          colours below.
        </p>
        <p>The search covers three sets:</p>
        <dl className="help-list">
          <dt>Uploaded</dt>
          <dd>Papers you added under Papers.</dd>
          <dt>Curated literature</dt>
          <dd>
            Papers that each compare two diseases, with the respects in which they are similar. These are the papers
            listed under Papers.
          </dd>
          <dt>Newly acquired</dt>
          <dd>
            The latest search of Europe PMC (October 2026). Most of these papers have only an abstract, and none are
            reviewed yet. Where a paper's open full text names two diseases in one passage, that pair is shown, but only
            whether the graph has an edge for it is checked.
          </dd>
        </dl>
      </>
    ),
  },
  yourDiseases: {
    title: 'Your diseases',
    body: (
      <>
        <p>
          Describe a disease that is not in the graph, for example a newly reported one, with a form or a JSON file.
          Every field except the name is optional.
        </p>
        <p>
          The v2 model compares it with every disease in the graph and draws dotted edges to the most similar ones. It
          uses whichever of its 12 kinds of evidence you provide: phenotypes, genes, drugs, inheritance, description and
          so on.
        </p>
        <p>Added diseases are saved in this browser only and are not shared with anyone.</p>
      </>
    ),
  },
  linksPerDisease: {
    title: 'Links shown per added disease',
    body: <p>How many of each added disease's closest matches are drawn as edges in the graph. Its details list them all.</p>,
  },
  papers: {
    title: 'Papers',
    body: (
      <>
        <p>
          Upload a paper to check the disease relationships it describes against the graph. Each claim, such as "disease A
          and disease B share a phenotype", is drawn on the graph and checked:
        </p>
        {verdictList}
        <p>
          For an uploaded paper, MedGemma (running on this computer) extracts the profile of the disease it describes, each
          feature backed by a quote from the paper, and the v2 model places that disease in the graph. The profile is then
          checked against its closest diseases.
        </p>
      </>
    ),
  },
  literatureSet: {
    title: 'Curated literature',
    body: (
      <p>
        A curated set of published papers that each compare two diseases. Each claim is checked against the graph in
        the same way as an uploaded paper. The summary shows how many of the relationships the graph recovers.
      </p>
    ),
  },
  filters: {
    title: 'Filters',
    body: (
      <p>
        Choose which edges are shown in the graph and the table. Filters never change the scores; they only hide edges.
        Reset restores the defaults.
      </p>
    ),
  },
  support: {
    title: 'Edge support',
    body: (
      <>
        <p>How well-established the link between two diseases is, independent of how high the similarity score is:</p>
        <dl className="help-list">
          <dt>
            <span className="line-key support-curated" aria-hidden /> Curated
          </dt>
          <dd>
            A recorded relation backs it: the diseases sit together in the Orphanet classification, share a causal gene
            (Orphanet), or share a drug trialled at phase 2 or later for both (Open Targets).
          </dd>
          <dt>
            <span className="line-key support-plausible" aria-hidden /> Plausible
          </dt>
          <dd>No recorded relation, but they share a broader Orphanet group, a gene, or any drug.</dd>
          <dt>
            <span className="line-key support-novel" aria-hidden /> Novel
          </dt>
          <dd>
            None of these. The model finds them similar from other evidence, such as phenotypes or text. These are
            hypotheses for expert review, and the most interesting leads for new connections.
          </dd>
        </dl>
        <p>The numbers show how many edges of each kind the graph has.</p>
      </>
    ),
  },
  mutual: {
    title: 'Mutual neighbours only',
    body: (
      <>
        <p>
          Each disease links to its 10 most similar diseases. An edge is <strong>mutual</strong> when both diseases have
          each other in their top 10, which makes it a stronger link.
        </p>
        <p>
          Turn this off to also show one-way edges. That is about twice as many, and the graph gets busier. Diseases
          with no mutual edge stay on the map as hollow, unconnected dots.
        </p>
      </>
    ),
  },
  score: {
    title: 'Minimum score',
    body: (
      <>
        <p>
          The similarity score the model gives each link: higher means the two diseases are more alike. It adds up the
          evidence from every kind of data the two diseases share, so it has no fixed maximum.
        </p>
        <p>
          In this graph scores run from about −3 to 30. A typical link scores about 2.5, and a typical mutual link about
          4. Raise the minimum to keep only the strongest similarities.
        </p>
      </>
    ),
  },
  mainEvidence: {
    title: 'Main evidence',
    body: (
      <>
        <p>
          The kind of evidence that contributes most to an edge's score. The model weighs 12 kinds: phenotypes, genes,
          pathways, Open Targets genes, drugs, drug targets, classification, name terms, description text, inheritance,
          age of onset and prevalence.
        </p>
        <p>
          For drug repurposing leads, choose <strong>Drugs</strong> or <strong>Drug targets</strong>, ideally with
          Novel support.
        </p>
      </>
    ),
  },
  cluster: {
    title: 'Cluster',
    body: (
      <p>
        Groups of diseases that are more similar to each other than to the rest, found automatically from the graph by
        community detection (Louvain). Each is named after the Orphanet category most common in it. Shows only edges
        with both diseases in the chosen cluster.
      </p>
    ),
  },
  category: {
    title: 'Orphanet category',
    body: (
      <p>
        Orphanet's top-level classification, such as rare neurological or rare skin disease. Shows only edges with both
        diseases in the chosen category. Unlike clusters, categories come from Orphanet rather than from the model.
      </p>
    ),
  },
  symptoms: {
    title: 'Filter by symptoms',
    body: (
      <>
        <p>
          Limit the graph to diseases with chosen symptoms (HPO terms). A symptom also matches its more specific forms:
          "Seizure" includes focal seizures. With several symptoms, choose whether a disease needs all of them or any.
        </p>
        <p>
          The links between the remaining diseases still come from the similarity model. Symptoms come from the
          Orphanet and HPO annotations, which cover about three-quarters of the diseases. A disease with no recorded
          symptoms never matches, but that does not mean it lacks the symptom.
        </p>
      </>
    ),
  },
  nodeColour: {
    title: 'Node colour',
    body: (
      <>
        <p>What the colour of each disease shows. It does not change the edges.</p>
        <dl className="help-list">
          <dt>Cluster</dt>
          <dd>
            Groups of diseases that link to each other far more than to the rest of the graph, found automatically by
            community detection (the Louvain method). Each is named after its most common Orphanet category.
          </dd>
          <dt>Orphanet category</dt>
          <dd>Orphanet's own top-level classification of the disease.</dd>
          <dt>Gene / Symptom</dt>
          <dd>Highlights the diseases linked to one gene, or with one symptom (HPO term, including its subtypes).</dd>
          <dt>Age of onset / Inheritance</dt>
          <dd>Colours diseases by when they start or how they are inherited.</dd>
          <dt>Similarity to a disease</dt>
          <dd>Shades every disease by how similar it is to the one you pick.</dd>
        </dl>
      </>
    ),
  },
  diseasePapers: {
    title: 'Papers in the literature set',
    body: (
      <>
        <p>
          Papers from this project's literature set that discuss this disease together with another one. Click a
          disease name to open that pair, or show the paper's claims on the graph.
        </p>
        <dl className="help-list">
          <dt>Curated</dt>
          <dd>From the reviewed set: each claim was read and scored for which kinds of similarity it describes.</dd>
          <dt>Unreviewed co-mention</dt>
          <dd>From the latest automatic acquisition run: the paper mentions both diseases, but nobody has checked the claim yet.</dd>
        </dl>
      </>
    ),
  },
  readingGraph: {
    title: 'Reading the graph',
    body: (
      <p>
        Each dot is a disease and each line links two similar diseases. Diseases with similar overall profiles sit close
        together. Larger dots have more links; hollow dots have no link under the current filters. Click a dot or a line
        to see why the diseases are considered similar.
      </p>
    ),
  },
}
