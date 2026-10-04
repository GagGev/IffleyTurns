import type {
  DatasetKey,
  Dimension,
  Disease,
  Edge,
  EdgeAggregate,
  LiteratureGraph,
  Paper,
  Relationship,
} from '../data/types'
import { RELATIONSHIPS } from '../data/types'

export type Measure = 'overall' | Dimension
export type Rarity = 'any' | 'one-rare' | 'both-rare'

export interface Filters {
  datasets: DatasetKey[]
  relationships: Relationship[]
  /** Which score drives thresholding, edge opacity and ranking. */
  measure: Measure
  minScore: number
  minPapers: number
  /** Minimum edge evidence weight (0–1). */
  minEvidence: number
  rarity: Rarity
}

export const DEFAULT_FILTERS: Filters = {
  datasets: ['review', 'pairfirst'],
  relationships: [...RELATIONSHIPS],
  measure: 'overall',
  minScore: 0,
  minPapers: 1,
  minEvidence: 0,
  rarity: 'any',
}

/** An edge as seen through the current dataset selection. */
export interface EdgeView {
  edge: Edge
  agg: EdgeAggregate
  /** Score for the selected measure, 0–1. */
  value: number
  papers: Paper[]
}

export function aggregateFor(edge: Edge, datasets: DatasetKey[]): EdgeAggregate | null {
  if (datasets.length === 0) return null
  if (datasets.length > 1) return edge.aggregates.all
  return edge.aggregates[datasets[0]] ?? null
}

export function measureValue(agg: EdgeAggregate, measure: Measure): number | null {
  return measure === 'overall' ? agg.similarity : (agg.dimensions[measure]?.score ?? null)
}

export function viewEdge(edge: Edge, filters: Pick<Filters, 'datasets' | 'measure'>): EdgeView | null {
  const agg = aggregateFor(edge, filters.datasets)
  if (!agg) return null
  const value = measureValue(agg, filters.measure)
  if (value === null) return null
  const papers = edge.papers.filter((p) => filters.datasets.includes(p.source))
  return { edge, agg, value, papers }
}

const isRare = (d: Disease | undefined) => d?.status === 'rare' || d?.status === 'uncertain'

export function filterEdges(
  graph: LiteratureGraph,
  filters: Filters,
  diseases: Map<string, Disease>,
): EdgeView[] {
  const out: EdgeView[] = []
  for (const edge of graph.edges) {
    const view = viewEdge(edge, filters)
    if (!view) continue
    const { agg, value } = view
    if (!filters.relationships.includes(agg.relationship)) continue
    if (value < filters.minScore) continue
    if (agg.nPapers < filters.minPapers) continue
    if (agg.evidenceWeight < filters.minEvidence) continue
    if (filters.rarity !== 'any') {
      const rareCount = Number(isRare(diseases.get(edge.source))) + Number(isRare(diseases.get(edge.target)))
      if (filters.rarity === 'one-rare' && rareCount < 1) continue
      if (filters.rarity === 'both-rare' && rareCount < 2) continue
    }
    out.push(view)
  }
  return out
}

/** Disease IDs within `depth` hops of any centre, following the given links. */
export function neighbourhood(
  links: { source: string; target: string }[],
  centres: string[],
  depth: number,
): Set<string> {
  const adjacency = new Map<string, string[]>()
  for (const { source, target } of links) {
    adjacency.set(source, [...(adjacency.get(source) ?? []), target])
    adjacency.set(target, [...(adjacency.get(target) ?? []), source])
  }
  const seen = new Set(centres)
  let frontier = [...centres]
  for (let d = 0; d < depth; d++) {
    const next: string[] = []
    for (const id of frontier) {
      for (const n of adjacency.get(id) ?? []) {
        if (!seen.has(n)) {
          seen.add(n)
          next.push(n)
        }
      }
    }
    frontier = next
  }
  return seen
}

export function pairId(a: string, b: string): string {
  return a < b ? `${a}|${b}` : `${b}|${a}`
}

export function otherEnd(edge: Edge, id: string): string {
  return edge.source === id ? edge.target : edge.source
}

/** Lower-cased, accent-free text for search matching. */
export function searchKey(value: string): string {
  return value
    .normalize('NFKD')
    .replace(/[̀-ͯ]/g, '')
    .toLowerCase()
}
