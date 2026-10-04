// Turning an uploaded paper into a v2_5 result.
//
// The real pipeline is `python -m v2_5.place_paper --input paper.pdf --output result.json` (MedGemma
// extraction, then v2 placement).  Until it is served to the browser, `mockExtractPaper` stands in: it
// returns a result in exactly the shape v2_5 writes, built from the graph itself, and marks it `mock`.
// A result file written by v2_5 can be uploaded directly and is used as is (`parsePaperResult`).
//
// To connect the real pipeline, replace `mockExtractPaper` with a call to an endpoint that runs v2_5
// and returns its JSON; nothing else in the UI changes.

import { loadAnnotations, type LoadedAnnotations } from '../data/annotations'
import { loadEdgeDetail } from '../data/source'
import type {
  DiseaseInput,
  GraphData,
  GraphEdge,
  GraphNode,
  PaperClaim,
  PaperEntry,
  PaperEvidenceItem,
  PaperResult,
  Placement,
  PlacementNeighbour,
} from '../data/types'
import { DIMENSION_MODALITIES } from './paperEval'
import { displayName } from './format'
import { otherEnd } from './graph'

const MODALITY_DIMENSION: Record<string, string> = Object.fromEntries(
  Object.entries(DIMENSION_MODALITIES).flatMap(([dimension, modalities]) => modalities.map((m) => [m, dimension])),
)

export const ACCEPTED_FILES = '.pdf,.txt,.md,.xml,.nxml,.json'

export interface PaperContext {
  graph: GraphData
  /** Edges of the v2 graph per disease (not the added ones). */
  edgesByNode: Map<string, GraphEdge[]>
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

function hash(text: string): number {
  let h = 5381
  for (let i = 0; i < text.length; i++) h = (Math.imul(h, 33) + text.charCodeAt(i)) >>> 0
  return h
}

// --- A v2_5 result file ---------------------------------------------------------------------

/** Reads a file written by `v2_5.place_paper --output`; returns an error message when it is something else. */
export function parsePaperResult(text: string): PaperResult | string {
  let data: unknown
  try {
    data = JSON.parse(text)
  } catch {
    return 'not-json'
  }
  const d = data as Partial<PaperResult> | null
  if (!d || typeof d !== 'object' || !Array.isArray(d.neighbors) || !d.record) return 'not-a-result'
  const name = d.display_name || d.focal_disease_label || d.provenance?.title || 'Uploaded paper'
  return {
    query_id: d.query_id ?? `USER:${hash(name).toString(36)}`,
    display_name: name,
    focal_disease_label: d.focal_disease_label ?? '',
    status: d.status ?? 'unknown',
    model: d.model ?? 'unknown',
    warnings: d.warnings ?? [],
    provenance: d.provenance ?? {},
    record: d.record,
    accepted_evidence: d.accepted_evidence ?? [],
    rejected_features: d.rejected_features ?? [],
    passages: d.passages,
    neighbors: d.neighbors,
    claims: d.claims,
  }
}

// --- The mock --------------------------------------------------------------------------------

async function headOf(file: File): Promise<{ text: string; title: string }> {
  const stem = file.name.replace(/\.[^.]+$/, '').replace(/[_-]+/g, ' ').trim()
  if (/\.(pdf)$/i.test(file.name)) return { text: '', title: stem }
  const text = (await file.text()).slice(0, 60_000)
  if (/\.json$/i.test(file.name)) {
    try {
      const json = JSON.parse(text.length < 60_000 ? text : '{}')
      return { text: `${json.title ?? ''} ${json.abstract ?? ''} ${json.body ?? json.text ?? ''}`, title: json.title || stem }
    } catch {
      return { text, title: stem }
    }
  }
  const firstLine = text.split(/\r?\n/).map((l) => l.trim()).find((l) => l.length > 12 && l.length < 200)
  return { text, title: firstLine?.replace(/<[^>]+>/g, '') || stem }
}

/** The catalogue disease the paper most plausibly concerns: its name appears in the file, else a stable pick. */
function chooseBase(graph: GraphData, haystack: string, seed: number): GraphNode {
  const hay = haystack.toLowerCase()
  let best: GraphNode | null = null
  for (const n of graph.nodes) {
    if (n.degree < 3 || n.name.length < 7) continue
    if (hay.includes(n.name.toLowerCase()) && (!best || n.name.length > best.name.length)) best = n
  }
  if (best) return best
  const pool = graph.nodes.filter((n) => n.degree >= 8)
  return pool[seed % pool.length]
}

const has = (sorted: number[], value: number) => {
  let lo = 0
  let hi = sorted.length - 1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    if (sorted[mid] === value) return true
    if (sorted[mid] < value) lo = mid + 1
    else hi = mid - 1
  }
  return false
}

function profileOf(ann: LoadedAnnotations, index: number, seed: number) {
  const genes = Object.entries(ann.raw.genes)
    .filter(([, n]) => has(n, index))
    .map(([g]) => g)
    .slice(0, 3)
  const phenotypes = Object.entries(ann.raw.phenotypes)
    .map(([id, p]) => ({ id, label: p.l, count: p.n.length, k: p.n.indexOf(index) >= 0 ? p.n.indexOf(index) : -1, f: p.f }))
    // Specific terms (carried by a few dozen diseases) read as extracted findings; very common ones as noise.
    .filter((p) => p.k >= 0 && p.count >= 3 && p.count <= 400 && p.f[p.k] >= 5)
    .sort((a, b) => hash(a.id + seed) - hash(b.id + seed))
    .slice(0, 8)
    .map((p) => ({ id: p.id, label: p.label, weight: p.f[p.k] / 10 }))
  const first = (source: Record<string, number[]>) =>
    Object.entries(source).find(([, n]) => has(n, index))?.[0] ?? null
  return { genes, phenotypes, onset: first(ann.raw.onset), inheritance: first(ann.raw.inheritance) }
}

const stagesMs = 450

export async function mockExtractPaper(
  file: File,
  ctx: PaperContext,
  onStage: (stage: string) => void,
): Promise<PaperResult> {
  const { graph, edgesByNode } = ctx
  onStage('Reading the paper')
  const [{ text, title }, ann] = await Promise.all([headOf(file), loadAnnotations().catch(() => null)])
  const seed = hash(`${file.name}:${file.size}`)
  const base = chooseBase(graph, `${file.name} ${text}`, seed)
  await sleep(stagesMs)
  onStage('Selecting passages (mock)')
  await sleep(stagesMs)
  onStage('Extracting a disease profile (mock MedGemma)')

  const index = ann?.index.get(base.id)
  const profile = ann && index !== undefined ? profileOf(ann, index, seed) : { genes: [], phenotypes: [], onset: null, inheritance: null }
  await sleep(stagesMs)
  onStage('Placing it in the v2 graph (mock)')

  const ranked = (edgesByNode.get(base.id) ?? []).slice().sort((a, b) => b.score - a.score).slice(0, 12)
  const neighbours: PlacementNeighbour[] = []
  for (const [rank, edge] of ranked.entries()) {
    const detail = await loadEdgeDetail(graph, edge.id)
    const other = graph.nodes.find((n) => n.id === otherEnd(edge, base.id))
    if (!detail || !other) continue
    neighbours.push({
      rank: rank + 1,
      id: other.id,
      name: other.name,
      category: other.category,
      score: edge.score,
      percentile: edge.percentile,
      explanation: detail.evidence.map((e) => ({
        modality: e.modality,
        similarity: detail.similarities[e.modality] ?? 0,
        contribution: e.contribution,
        shared: e.shared,
      })),
      known_relations: detail.relations as Record<string, string>,
      similarities: Object.fromEntries(Object.entries(detail.similarities).filter(([, v]) => v !== null)) as Record<string, number>,
      contributions: detail.contributions,
      annotation_adjustment: detail.annotationAdjustment,
      support: edge.support,
    })
  }
  const details = new Map(neighbours.map((n) => [n.id, n]))

  // The paper's own comparisons: what it says, and for some of them something the graph does not carry.
  const claims: PaperClaim[] = []
  const mainDimension = (n: PlacementNeighbour) => MODALITY_DIMENSION[n.explanation[0]?.modality ?? ''] ?? 'phenotype'
  const weakDimension = (n: PlacementNeighbour) =>
    ['pathways', 'treatment', 'genes', 'phenotype'].find((d) =>
      (DIMENSION_MODALITIES[d] ?? []).every((m) => (n.contributions[m] ?? 0) < 0.25),
    ) ?? 'pathways'
  const quote = (a: string, b: string, how: string) => `${a} and ${b} ${how} [mock quote]`
  const add = (n: PlacementNeighbour, dimensions: string[], relationship: string, how: string, feature?: string) => {
    const section = ['Results', 'Discussion', 'Introduction'][claims.length % 3]
    claims.push({
      a: '@focal',
      b: n.id,
      dimensions,
      relationship,
      stated: relationship === 'similar' ? 0.75 : 0.5,
      finding: quote(displayName(base.name), displayName(n.name), how),
      feature,
      quote: quote(displayName(base.name), displayName(n.name), how),
      locator: `${section}:p${2 + claims.length * 3}`,
    })
  }
  const pick = (i: number) => neighbours[i]
  if (pick(0)) add(pick(0), [mainDimension(pick(0))], 'similar', `share ${mainDimension(pick(0))} features`, pick(0).explanation[0]?.shared[0])
  if (pick(1)) add(pick(1), [mainDimension(pick(1))], 'similar', `have overlapping ${mainDimension(pick(1))}`, pick(1).explanation[0]?.shared[0])
  if (pick(3)) add(pick(3), [mainDimension(pick(3)), weakDimension(pick(3))], 'related but distinct', `are related through ${mainDimension(pick(3))} and ${weakDimension(pick(3))}`)
  if (pick(5)) add(pick(5), [weakDimension(pick(5))], 'related but distinct', `are linked by ${weakDimension(pick(5))}`)
  const comparator = graph.nodes.find(
    (n) => n.category === base.category && n.id !== base.id && n.degree >= 6 && !details.has(n.id) && hash(n.id + seed) % 40 === 0,
  )
  if (comparator) {
    claims.push({
      a: '@focal',
      b: comparator.id,
      dimensions: ['phenotype'],
      relationship: 'similar',
      stated: 0.65,
      finding: quote(displayName(base.name), displayName(comparator.name), 'present with a similar clinical picture'),
      quote: quote(displayName(base.name), displayName(comparator.name), 'present with a similar clinical picture'),
      locator: 'Discussion:p9',
    })
  }

  const evidence: PaperEvidenceItem[] = []
  const item = (feature_type: string, identifier: string, label: string, snippet: string, confidence = 3) =>
    evidence.push({
      feature_type,
      identifier,
      label,
      locator: `${['Results', 'Abstract', 'Case report'][evidence.length % 3]}:p${1 + (evidence.length % 5)}`,
      quote: `${snippet} [mock quote]`,
      confidence,
      extraction_method: 'mock',
      verification_status: 'mock',
    })
  for (const gene of profile.genes) item('gene', gene, gene, `Pathogenic variants in ${gene} were identified in the proband`)
  for (const p of profile.phenotypes) item('phenotype', p.id, p.label, `The patient presented with ${p.label.toLowerCase()}`, p.weight >= 0.8 ? 3 : 2)
  if (profile.onset) item('onset', profile.onset, profile.onset, `Symptoms began in the ${profile.onset.toLowerCase()} period`, 2)
  if (profile.inheritance) item('inheritance', profile.inheritance, profile.inheritance, `The pedigree is consistent with ${profile.inheritance.toLowerCase()} transmission`, 2)

  const display = title.length > 70 ? `${title.slice(0, 67)}…` : title
  return {
    query_id: `USER:mock-${hash(file.name).toString(36)}`,
    display_name: display,
    focal_disease_label: `${base.name} (inferred)`,
    status: 'mock',
    model: 'mock extractor (no model was run)',
    warnings: [
      'Mock output: features, quotes and comparisons are generated from the graph, not read from the paper. Replace mockExtractPaper with the v2_5 pipeline.',
    ],
    provenance: { title, input_path: file.name },
    record: {
      name: '',
      description: `Mock profile for ${display}`,
      phenotypes: Object.fromEntries(profile.phenotypes.map((p) => [p.id, p.weight])),
      genes: profile.genes,
      onset: profile.onset ? [profile.onset] : [],
      inheritance: profile.inheritance ? [profile.inheritance] : [],
    },
    accepted_evidence: evidence,
    rejected_features: [],
    neighbors: neighbours,
    claims,
    mock: true,
  }
}

// --- Turning a result into app state -----------------------------------------------------------

/** The paper entry for a result whose focal disease was added to the user's diseases as `focalId`. */
export function paperEntryFor(result: PaperResult, focalId: string, fileName: string): PaperEntry {
  const claims = (result.claims ?? []).map((c) => ({ ...c, a: c.a === '@focal' || c.a === result.query_id ? focalId : c.a }))
  return {
    id: `upload:${focalId}`,
    title: result.provenance.title || result.display_name || fileName,
    year: result.provenance.year ?? null,
    link: result.provenance.doi ? `https://doi.org/${result.provenance.doi}` : undefined,
    kind: 'upload',
    mock: result.mock === true,
    claims,
    // The neighbours live on the placement; keep the rest of the extraction.
    result: { ...result, neighbors: [] },
    focalId,
  }
}

/** The disease described by the paper's extracted record, in v2's input format, plus readable names for its IDs. */
export function inputFromResult(result: PaperResult): { input: DiseaseInput; labels: Record<string, string> } {
  const r = result.record as DiseaseInput
  const labels = Object.fromEntries(
    result.accepted_evidence.filter((e) => e.feature_type === 'phenotype').map((e) => [e.identifier, e.label]),
  )
  return {
    input: {
      name: result.display_name,
      description: r.description,
      phenotypes: r.phenotypes,
      genes: r.genes,
      drugs: r.drugs,
      inheritance: r.inheritance,
      onset: r.onset,
      prevalence: r.prevalence,
      ontology_parents: r.ontology_parents,
    },
    labels,
  }
}

/** The placement the paper's neighbours amount to, as the placement service would return it. */
export function placementFromResult(result: PaperResult, id: string): Placement {
  const present = new Set<string>()
  for (const n of result.neighbors) for (const e of n.explanation) present.add(e.modality)
  const r = result.record as DiseaseInput
  if (Object.keys(r.phenotypes ?? {}).length > 0) present.add('phenotype')
  if ((r.genes ?? []).length > 0) present.add('gene')
  if ((r.drugs ?? []).length > 0) present.add('drug')
  if ((r.onset ?? []).length > 0) present.add('onset')
  if ((r.inheritance ?? []).length > 0) present.add('inheritance')
  if (r.prevalence) present.add('prevalence')
  if (r.description) present.add('text')
  return {
    id,
    name: result.display_name,
    present: [...present],
    warnings: result.warnings,
    weights: {},
    neighbours: result.neighbors,
    placedAt: new Date().toISOString(),
  }
}
