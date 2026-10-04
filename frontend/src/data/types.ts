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

export interface GraphNode {
  id: string
  name: string
  category: string
  disorderType: string
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
