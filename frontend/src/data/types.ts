// Shapes of the data the UI consumes.  The graph mirrors v2/.data/graph (via
// scripts/build_graph_data.py); placements mirror the response of
// api/server.py, which wraps v2's place_disease.place().

export type Support = 'curated' | 'plausible' | 'novel'
export const SUPPORT_LEVELS: Support[] = ['curated', 'plausible', 'novel']

export const SUPPORT_MEANING: Record<Support, string> = {
  curated: 'A curated relation backs this edge: Orphanet classification, a shared causal gene or a shared trial drug.',
  plausible: 'No curated relation, but the diseases share an Orphanet group, a curated gene or a drug.',
  novel: 'None of these: a hypothesis for expert review.',
}

/** Short labels for v2's 12 modalities; descriptions come from the graph file. */
export const MODALITY_LABELS: Record<string, string> = {
  phenotype: 'Phenotypes',
  gene: 'Genes',
  pathway: 'Pathways',
  ot_gene: 'Open Targets genes',
  drug: 'Drugs',
  drug_target: 'Drug targets',
  ontology: 'Classification',
  name: 'Name terms',
  text: 'Description text',
  inheritance: 'Inheritance',
  onset: 'Age of onset',
  prevalence: 'Prevalence',
}

export type NodeOrigin = 'catalogue' | 'shared' | 'user'

/** What the nodes of the graph are coloured by. */
export type ColourBy =
  | 'cluster'
  | 'category'
  | 'gene'
  | 'symptom'
  | 'onset'
  | 'inheritance'
  | 'disease'
  | 'plain'

/** One Louvain community of the similarity graph (computed by scripts/build_graph_data.py). */
export interface ClusterInfo {
  /** "C1", "C2", ...: numbered by size, so C1 is the largest. */
  id: string
  /** Dominant Orphanet category followed by distinctive name terms. */
  label: string
  size: number
  topCategory: string
  /** Share of members in the dominant category. */
  purity: number
  categories: [string, number][]
  terms: string[]
  hubId: string
  hubName: string
}

export interface ClusteringInfo {
  method: string
  resolution: number
  edgeWeight: string
  modularity: number
  clusters: number
}

export interface GraphNode {
  id: string
  name: string
  category: string
  disorderType: string
  /** Index into GraphData.clusters, or -1 (not clustered, or an added disease with no neighbours). */
  cluster: number
  /** Modalities annotated for this disease. */
  modalities: string[]
  degree: number
  /** catalogue: Orphanet; shared: added to the v2 graph with place_disease.py --add; user: added in this browser. */
  origin: NodeOrigin
  x?: number
  y?: number
}

export interface GraphEdge {
  id: string
  source: string
  target: string
  /** Fusion logit; higher is more similar. */
  score: number
  /** Share of random catalogue pairs scoring lower. */
  percentile: number
  support: Support
  /** Both diseases list each other among their top-k neighbours. */
  mutual: boolean
  /** Modality contributing most to the score. */
  mainModality: string | null
  origin: 'graph' | 'user'
}

export interface Evidence {
  modality: string
  contribution: number
  similarity?: number
  shared: string[]
}

export interface EdgeDetail {
  /** Cosine similarity per modality; null when not annotated for both diseases. */
  similarities: Record<string, number | null>
  contributions: Record<string, number>
  evidence: Evidence[]
  relations: { orphanet?: string; gene?: string; drug?: string }
  annotationAdjustment: number
  /** Rank of target among source's neighbours, and the reverse. */
  ranks: [number | null, number | null]
}

export interface GraphData {
  generatedAt: string
  k: number | null
  modalities: string[]
  modalityDescriptions: Record<string, string>
  categories: string[]
  clusters: ClusterInfo[]
  clustering: ClusteringInfo | null
  hasLayout: boolean
  shards: number
  stats: { nodes: number; edges: number; mutualEdges: number; support: Partial<Record<Support, number>> }
  nodes: GraphNode[]
  edges: GraphEdge[]
}

/** A new disease in v2's input format (v2/README.md, "Placing a new disease"). */
export interface DiseaseInput {
  name: string
  synonyms?: string[]
  description?: string
  /** HPO term → share of patients affected (lists default to 0.5 in v2). */
  phenotypes?: Record<string, number> | string[]
  genes?: string[]
  drugs?: string[]
  inheritance?: string[]
  onset?: string[]
  /** Fraction of people affected, e.g. 1e-5 for 1 in 100,000. */
  prevalence?: number
  ontology_parents?: string[]
}

export interface PlacementNeighbour {
  rank: number
  id: string
  name: string
  category: string
  score: number
  percentile: number
  explanation: { modality: string; similarity: number; contribution: number; shared: string[] }[]
  known_relations: Record<string, string>
  similarities: Record<string, number>
  contributions: Record<string, number>
  annotation_adjustment: number
  support: Support
}

export interface Placement {
  id: string
  name: string
  /** Modalities the new disease could be encoded in. */
  present: string[]
  warnings: string[]
  /** Fusion weights refitted for those modalities. */
  weights: Record<string, number>
  neighbours: PlacementNeighbour[]
  placedAt: string
}

export interface ApiHealth {
  status: 'loading' | 'ready' | 'error' | 'offline'
  error?: string
  diseases?: number
}

// --- Annotations (public/data/annotations.json) ----------------------------------------

/** Which diseases carry each gene, HPO term, onset class and inheritance mode (node indices of graph.json). */
export interface Annotations {
  nodes: string[]
  genes: Record<string, number[]>
  /** `l` label, `n` carrying diseases, `f` share of patients in tenths (0-10), parallel to `n`. */
  phenotypes: Record<string, { l: string; n: number[]; f: number[] }>
  onset: Record<string, number[]>
  inheritance: Record<string, number[]>
}

// --- Papers --------------------------------------------------------------------------------

/** A paper's statement that two diseases are similar, and in which respects. */
export interface PaperClaim {
  a: string
  b: string
  /** Dimensions the paper names: phenotype, genes, pathways, treatment, epidemiology, other (...). */
  dimensions: string[]
  relationship: string
  /** Similarity the paper's reading was scored at (0-1), when known. */
  stated: number | null
  finding: string
  /** The shared feature the paper names, when the extractor found one (gene symbol, HPO label ...). */
  feature?: string
  quote?: string
  locator?: string
  lowEvidence?: boolean
  /** Literature set only: ID of the graph's edge for this pair and its explanation, baked in at build time. */
  edge?: string
  detail?: RawEdgeDetail
}

/** The compact explanation of an edge as stored in details/NN.json. */
export interface RawEdgeDetail {
  s: (number | null)[]
  c: number[]
  e: [number, number, string[]][]
  r: EdgeDetail['relations']
  a: number
  k: [number | null, number | null]
}

/**
 * One evidence item of a v2_5 extraction (v2_5/extraction.py AcceptedEvidence).
 */
export interface PaperEvidenceItem {
  feature_type: string
  identifier: string
  label: string
  locator: string
  quote: string
  confidence: number
  extraction_method: string
  verification_status: string
}

/** The output of `python -m v2_5.place_paper --output ...`, plus the optional `claims` list. */
export interface PaperResult {
  query_id: string
  display_name: string
  focal_disease_label: string
  status: string
  model: string
  warnings: string[]
  provenance: { title?: string; year?: number; doi?: string; pmid?: string | number; input_path?: string }
  record: DiseaseInput & Record<string, unknown>
  accepted_evidence: PaperEvidenceItem[]
  rejected_features: { feature_type: string; value: string; reason: string }[]
  passages?: { word_count: number }[]
  neighbors: PlacementNeighbour[]
  /** Comparisons the paper itself draws between the focal disease and others (not produced by v2_5 today). */
  claims?: PaperClaim[]
  /** True for output produced by the mock extractor in the browser. */
  mock?: boolean
}

/** A paper the explorer can check against the graph. */
export interface PaperEntry {
  id: string
  title: string
  year: number | null
  link?: string
  kind: 'literature' | 'upload'
  mock: boolean
  claims: PaperClaim[]
  /** For uploads: the v2_5 output, and the placed disease that represents the paper's focal disease. */
  result?: PaperResult
  focalId?: string
}

export type ClaimVerdict = 'confirmed' | 'partial' | 'unsupported' | 'missing' | 'edge-only'
