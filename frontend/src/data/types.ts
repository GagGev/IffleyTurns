// Shapes of the data the UI consumes.  The literature graph mirrors the JSON
// written by scripts/build_literature_graph.py; the computed-similarity types
// mirror the result dict returned by evaluation.py's DiseaseDistanceEvaluator.

export const DIMENSIONS = [
  'phenotype',
  'genetic',
  'mechanism',
  'therapeutic',
  'natural_history',
  'diagnostic_confusability',
  'comorbidity',
] as const

export type Dimension = (typeof DIMENSIONS)[number]

export const DIMENSION_LABELS: Record<Dimension, string> = {
  phenotype: 'Phenotype',
  genetic: 'Genetic',
  mechanism: 'Mechanism',
  therapeutic: 'Therapeutic',
  natural_history: 'Natural history',
  diagnostic_confusability: 'Diagnostic confusability',
  comorbidity: 'Comorbidity',
}

export type Relationship = 'similar' | 'related but distinct' | 'unrelated'
export const RELATIONSHIPS: Relationship[] = ['similar', 'related but distinct', 'unrelated']

export type DatasetKey = 'review' | 'pairfirst'
export type RarityStatus = 'rare' | 'common' | 'uncertain' | 'unknown'

export interface Disease {
  id: string
  name: string
  aliases: string[]
  orphaId: string | null
  candidateOrphaIds: string[]
  status: RarityStatus
  degree: number
  /**
   * Where the node comes from: the literature graph (default), a disease the
   * user added, or a catalogue disease pulled in as a computed neighbour.
   */
  origin?: 'literature' | 'user' | 'catalogue'
}

/** A feature-based similarity link from a user-added disease. */
export interface ComputedEdge {
  id: string
  source: string
  target: string
  similarity: number
}

export interface PaperDimension {
  w: number
  s: number | null
}

export interface Paper {
  pmid: string
  title: string
  year: number | null
  link: string
  studyType: string
  design: string
  sampleSize: number | null
  relationship: Relationship
  similarity: number | null
  dimensionLabel: string
  finding: string
  keyDetails: string
  evidenceChecked: string
  openAccess: string
  source: DatasetKey
  batch: string
  stratum: string
  evidenceScore: number
  dimensions: Partial<Record<Dimension, PaperDimension>>
}

export interface DimensionConsensus {
  score: number
  weight: number
  papers: number
}

/** Edge-level evidence for one selection of datasets. */
export interface EdgeAggregate {
  /** Evidence-weighted mean of paper similarity scores, 0–1. */
  similarity: number | null
  /** Best paper score plus replication bonus (0–15), as in build_edge_ranking.py. */
  evidenceScore: number
  /** evidenceScore / 15. */
  evidenceWeight: number
  nPapers: number
  relationship: Relationship
  relationshipCounts: Partial<Record<Relationship, number>>
  bestDesign: string
  flags: string[]
  dimensions: Partial<Record<Dimension, DimensionConsensus>>
}

export interface Edge {
  id: string
  source: string
  target: string
  datasets: DatasetKey[]
  strata: string[]
  aggregates: { all: EdgeAggregate } & Partial<Record<DatasetKey, EdgeAggregate>>
  papers: Paper[]
}

export interface LiteratureGraph {
  schemaVersion: number
  generatedAt: string
  dimensions: Dimension[]
  datasets: { key: DatasetKey; label: string; files: string[] }[]
  stats: { paperRows: number; selfPairsSkipped: number; nodes: number; edges: number }
  nodes: Disease[]
  edges: Edge[]
}

/** One feature family's contribution, as produced by evaluation.py. */
export interface FeatureComponent {
  feature: string
  label: string
  available: boolean
  reason?: string
  similarity?: number
  configured_weight: number
  effective_weight: number
  shared?: string[]
  shared_count?: number
  union_count?: number
  left_count?: number
  right_count?: number
}

/** DiseaseDistanceEvaluator.calculate_distance() result. */
export interface ComputedSimilarity {
  disease_a: { orpha_id: string; name: string }
  disease_b: { orpha_id: string; name: string }
  similarity: number | null
  distance: number | null
  comparable: boolean
  components: FeatureComponent[]
  interpretation: string
}
