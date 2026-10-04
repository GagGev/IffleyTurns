// The boundary between the UI and its data: the static v2 graph files written
// by scripts/build_graph_data.py, and the placement API in api/server.py.

import type {
  ApiHealth,
  DiseaseInput,
  EdgeDetail,
  GraphData,
  GraphEdge,
  GraphNode,
  Placement,
  PlacementNeighbour,
  RawEdgeDetail,
  Support,
} from './types'
import { SUPPORT_LEVELS } from './types'

const dataUrl = (file: string) => `${import.meta.env.BASE_URL}data/${file}`

export class GraphNotBuiltError extends Error {}

interface RawGraph {
  generatedAt: string
  k: number | null
  modalities: string[]
  modalityDescriptions: Record<string, string>
  categories: string[]
  clusters?: GraphData['clusters']
  clustering?: GraphData['clustering']
  hasLayout: boolean
  shards: number
  stats: GraphData['stats']
  nodes: { id: string; name: string; c: number; k?: number; t: string; m: number; u?: number; x?: number; y?: number }[]
  /** [source, target, score, percentile, support, mutual, main modality] */
  edges: [number, number, number, number, number, number, number][]
}

export async function loadGraph(): Promise<GraphData> {
  const response = await fetch(dataUrl('graph.json'))
  // The dev server answers missing files with index.html, so check the type too.
  if (!response.ok || !response.headers.get('content-type')?.includes('json')) {
    throw new GraphNotBuiltError('graph.json not found')
  }
  const raw: RawGraph = await response.json()
  const degree = new Array(raw.nodes.length).fill(0)
  for (const [s, t] of raw.edges) {
    degree[s]++
    degree[t]++
  }
  const nodes: GraphNode[] = raw.nodes.map((n, i) => ({
    id: n.id,
    name: n.name,
    category: raw.categories[n.c] ?? '',
    disorderType: n.t,
    cluster: n.k ?? -1,
    modalities: raw.modalities.filter((_, b) => n.m & (1 << b)),
    degree: degree[i],
    origin: n.u ? 'shared' : 'catalogue',
    x: n.x,
    y: n.y,
  }))
  const edges: GraphEdge[] = raw.edges.map(([s, t, score, percentile, support, mutual, main]) => ({
    id: `${nodes[s].id}|${nodes[t].id}`,
    source: nodes[s].id,
    target: nodes[t].id,
    score,
    percentile,
    support: SUPPORT_LEVELS[support] ?? 'novel',
    mutual: mutual === 1,
    mainModality: raw.modalities[main] ?? null,
    origin: 'graph',
  }))
  return { ...raw, clusters: raw.clusters ?? [], clustering: raw.clustering ?? null, nodes, edges }
}

// --- Edge explanations, sharded ---------------------------------------------------

/** djb2, identical to edge_shard() in scripts/build_graph_data.py. */
export function edgeShard(edgeId: string, shards: number): number {
  let h = 5381
  for (let i = 0; i < edgeId.length; i++) h = (Math.imul(h, 33) + edgeId.charCodeAt(i)) >>> 0
  return h % shards
}

type RawDetail = RawEdgeDetail

/** Expand the compact form stored in details/NN.json and literature.json. */
export function expandDetail(modalities: string[], raw: RawDetail): EdgeDetail {
  return {
    similarities: Object.fromEntries(modalities.map((name, i) => [name, raw.s[i]])),
    contributions: Object.fromEntries(modalities.map((name, i) => [name, raw.c[i]])),
    evidence: raw.e.map(([i, contribution, shared]) => ({ modality: modalities[i], contribution, shared })),
    relations: raw.r,
    annotationAdjustment: raw.a,
    ranks: raw.k,
  }
}

const shardCache = new Map<number, Promise<Record<string, RawDetail>>>()

export async function loadEdgeDetail(graph: GraphData, edgeId: string): Promise<EdgeDetail | null> {
  const shard = edgeShard(edgeId, graph.shards)
  let pending = shardCache.get(shard)
  if (!pending) {
    pending = fetch(dataUrl(`details/${String(shard).padStart(2, '0')}.json`)).then((r) => {
      if (!r.ok) throw new Error(`Could not load edge details (${r.status})`)
      return r.json()
    })
    pending.catch(() => shardCache.delete(shard))
    shardCache.set(shard, pending)
  }
  const raw = (await pending)[edgeId]
  return raw ? expandDetail(graph.modalities, raw) : null
}

const RELATION_KEYS: Record<string, keyof EdgeDetail['relations']> = {
  orphanet_siblings: 'orphanet',
  shared_causal_gene: 'gene',
  shared_drug: 'drug',
}

/** The same explanation shape for an edge from a placed (user) disease. */
export function detailFromPlacement(n: PlacementNeighbour, modalities: string[]): EdgeDetail {
  return {
    similarities: Object.fromEntries(modalities.map((m) => [m, n.similarities[m] ?? null])),
    contributions: Object.fromEntries(modalities.map((m) => [m, n.contributions[m] ?? 0])),
    evidence: n.explanation,
    relations: Object.fromEntries(
      Object.entries(n.known_relations).map(([k, v]) => [RELATION_KEYS[k] ?? k, v]),
    ) as EdgeDetail['relations'],
    annotationAdjustment: n.annotation_adjustment,
    ranks: [n.rank, null],
  }
}

// --- Placement API ------------------------------------------------------------------

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${import.meta.env.BASE_URL}api/${path}`, init)
  } catch {
    throw new Error('The placement service is not running. Start it with `python frontend/api/server.py`.')
  }
  const type = response.headers.get('content-type') ?? ''
  if (!type.includes('json')) {
    throw new Error('The placement service is not running. Start it with `python frontend/api/server.py`.')
  }
  const body = await response.json()
  if (!response.ok) throw new Error(body.error ?? `Request failed (${response.status})`)
  return body as T
}

export async function apiHealth(): Promise<ApiHealth> {
  try {
    return await api<ApiHealth>('health')
  } catch {
    return { status: 'offline' }
  }
}

export async function placeDisease(
  id: string,
  disease: DiseaseInput,
  others: { id: string; disease: DiseaseInput }[],
  top = 20,
): Promise<Placement> {
  const result = await api<Omit<Placement, 'placedAt'>>('place', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id, disease, others, top }),
  })
  return { ...result, placedAt: new Date().toISOString() }
}

export type SuggestField = 'phenotypes' | 'genes' | 'drugs' | 'ontology'

export async function suggest(field: SuggestField, q: string): Promise<{ id: string; label: string }[]> {
  try {
    return await api(`suggest?field=${field}&q=${encodeURIComponent(q)}`)
  } catch {
    return []
  }
}

export function supportRank(s: Support): number {
  return SUPPORT_LEVELS.indexOf(s)
}
