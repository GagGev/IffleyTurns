import type { GraphEdge, GraphNode, Support } from '../data/types'
import { SUPPORT_LEVELS } from '../data/types'

export interface Filters {
  support: Support[]
  /** Minimum share of random pairs the edge must outscore. */
  minPercentile: number
  /** Only edges whose largest contribution comes from this modality. */
  mainModality: string | null
  category: string | null
  /** Only edges where both diseases list each other among their top-k. */
  mutualOnly: boolean
}

export const DEFAULT_FILTERS: Filters = {
  support: [...SUPPORT_LEVELS],
  minPercentile: 0,
  mainModality: null,
  category: null,
  mutualOnly: true,
}

export const PERCENTILE_STEPS = [0, 0.9, 0.99, 0.999, 0.9999]

export function filterEdges(edges: GraphEdge[], filters: Filters, nodes: Map<string, GraphNode>): GraphEdge[] {
  return edges.filter((e) => {
    if (!filters.support.includes(e.support)) return false
    if (e.percentile < filters.minPercentile) return false
    if (filters.mainModality && e.mainModality !== filters.mainModality) return false
    // Links from added diseases are never mutual; keep them visible.
    if (filters.mutualOnly && !e.mutual && e.origin === 'graph') return false
    if (filters.category) {
      const a = nodes.get(e.source)
      const b = nodes.get(e.target)
      const inside = (n?: GraphNode) => n?.category === filters.category || n?.origin === 'user'
      if (!inside(a) || !inside(b)) return false
    }
    return true
  })
}

/** Disease IDs within `depth` hops of any centre, following the given edges. */
export function neighbourhood(edges: { source: string; target: string }[], centres: string[], depth: number): Set<string> {
  const adjacency = new Map<string, string[]>()
  for (const { source, target } of edges) {
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

/** Edge IDs follow v2's edges.csv, which lists each pair once in its own order; look up both. */
export function findEdge(byId: Map<string, GraphEdge>, a: string, b: string): GraphEdge | undefined {
  return byId.get(`${a}|${b}`) ?? byId.get(`${b}|${a}`)
}

export function otherEnd(edge: GraphEdge, id: string): string {
  return edge.source === id ? edge.target : edge.source
}

/** Lower-cased, accent-free text for search matching. */
export function searchKey(value: string): string {
  return value
    .normalize('NFKD')
    .replace(/[̀-ͯ]/g, '')
    .toLowerCase()
}
