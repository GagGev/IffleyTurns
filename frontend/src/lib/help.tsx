// Definitions behind the "?" buttons in the sidebar.  Kept in one place so the
// wording can be reviewed and edited without touching the components.

import type { ReactNode } from 'react'
import { VERDICT_LABEL, VERDICT_MEANING, VERDICT_ORDER } from './paperEval'

export type HelpTopic =
  | 'search'
  | 'yourDiseases'
  | 'linksPerDisease'
  | 'papers'
  | 'literatureSet'
  | 'filters'
  | 'support'
  | 'mutual'
  | 'percentile'
  | 'mainEvidence'
  | 'cluster'
  | 'category'
  | 'nodeColour'
  | 'readingGraph'

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
        <p>Claim extraction is a mock until the paper pipeline is connected.</p>
      </>
    ),
  },
  literatureSet: {
    title: 'Literature set',
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
        <p>Turn this off to also show one-way edges. That is about twice as many, and the graph gets busier.</p>
      </>
    ),
  },
  percentile: {
    title: 'Minimum percentile',
    body: (
      <>
        <p>
          How unusual the similarity is. The model scored 200,000 random pairs of diseases. An edge at the 99.9th
          percentile scores higher than 99.9% of those random pairs.
        </p>
        <p>Raise this to keep only the most exceptional similarities.</p>
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
  nodeColour: {
    title: 'Node colour',
    body: (
      <>
        <p>What the colour of each disease shows. It does not change the edges.</p>
        <dl className="help-list">
          <dt>Cluster / Orphanet category</dt>
          <dd>Which group the disease belongs to.</dd>
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
  readingGraph: {
    title: 'Reading the graph',
    body: (
      <p>
        Each dot is a disease and each line links two similar diseases. Diseases with similar overall profiles sit close
        together. Larger dots have more links. Click a dot or a line to see why the diseases are considered similar.
      </p>
    ),
  },
}
