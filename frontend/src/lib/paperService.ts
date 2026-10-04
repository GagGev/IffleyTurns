// Turning an uploaded paper into a v2_5 result.
//
// `extractPaper` runs the real pipeline through the placement service (api/papers.py): the same steps as
// `python -m v2_5.place_paper --input paper.pdf --output result.json`, MedGemma extraction then v2 placement.
// A result file written by v2_5 can also be uploaded directly and is used as is (`parsePaperResult`).

import { api } from '../data/source'
import type { DiseaseInput, PaperEntry, PaperResult, Placement } from '../data/types'

export const ACCEPTED_FILES = '.pdf,.txt,.md,.xml,.nxml,.json'

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

// --- The v2_5 pipeline ------------------------------------------------------------------------

function base64Of(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result).split(',')[1] ?? '')
    reader.onerror = () => reject(new Error('The file could not be read.'))
    reader.readAsDataURL(file)
  })
}

/**
 * Runs v2_5 on the paper through the placement service: MedGemma extracts an evidence-grounded disease profile
 * (every feature quoted from the paper and checked against v2's vocabulary), then v2 places it in the graph.
 * MedGemma runs on this computer and takes a few minutes; `onStage` gets a running timer meanwhile.
 */
export async function extractPaper(file: File, onStage: (stage: string) => void): Promise<PaperResult> {
  onStage('Reading the paper')
  const content = await base64Of(file)
  const started = Date.now()
  const tick = () => {
    const s = Math.round((Date.now() - started) / 1000)
    onStage(`MedGemma is reading the paper and extracting the disease profile… ${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')} (usually 2–5 minutes)`)
  }
  tick()
  const timer = setInterval(tick, 1000)
  try {
    const result = await api<unknown>('papers/place', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename: file.name, content, top: 20 }),
    })
    const parsed = parsePaperResult(JSON.stringify(result))
    if (typeof parsed === 'string') throw new Error('The placement service returned an unexpected result.')
    return parsed
  } finally {
    clearInterval(timer)
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
