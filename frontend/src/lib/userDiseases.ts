// Diseases a researcher adds in the browser.  Each keeps its description in
// v2's input format (exactly what v2/place_disease.py --json accepts) and the
// last placement the API returned, so the graph still shows it when the API
// is not running.

import { useCallback, useState } from 'react'
import type { DiseaseInput, Placement } from '../data/types'

export interface UserDisease {
  id: string
  input: DiseaseInput
  /** Readable names for IDs picked from suggestions (HPO terms, drugs, ...). */
  labels?: Record<string, string>
  placement?: Placement
  createdAt: string
}

// v2 vocabularies (v2/modalities.py INHERITANCE_MAP, ONSET_BINS).
export const INHERITANCE_OPTIONS = [
  'Autosomal recessive',
  'Autosomal dominant',
  'Semidominant',
  'X-linked recessive',
  'X-linked dominant',
  'X-linked',
  'Y-linked',
  'Mitochondrial inheritance',
  'Multigenic/multifactorial',
  'Oligogenic',
  'Not applicable',
]
export const ONSET_OPTIONS = ['Antenatal', 'Neonatal', 'Infancy', 'Childhood', 'Adolescent', 'Adult', 'Elderly', 'All ages']

/** Share of patients with a phenotype; v2 uses the weight directly (lists default to 0.5). */
export const FREQUENCIES: { label: string; weight: number }[] = [
  { label: 'Always', weight: 1 },
  { label: 'Very frequent', weight: 0.9 },
  { label: 'Frequent', weight: 0.55 },
  { label: 'Occasional', weight: 0.17 },
  { label: 'Very rare', weight: 0.03 },
  { label: 'Unknown', weight: 0.5 },
]

/** Orphanet prevalence classes as v2-style fractions (class midpoints). */
export const PREVALENCE_CLASSES: { label: string; value: number }[] = [
  { label: '<1 / 1 000 000', value: 5e-7 },
  { label: '1-9 / 1 000 000', value: 5e-6 },
  { label: '1-9 / 100 000', value: 5e-5 },
  { label: '1-5 / 10 000', value: 3e-4 },
  { label: '6-9 / 10 000', value: 7.5e-4 },
  { label: '>1 / 1000', value: 1e-3 },
]

export const V2_FIELDS = new Set([
  'id',
  'name',
  'synonyms',
  'description',
  'phenotypes',
  'hpo_ids',
  'genes',
  'pathways',
  'ot_genes',
  'drugs',
  'inheritance',
  'onset',
  'prevalence',
  'ontology_parents',
  'parents',
])

export function normaliseHpo(value: string): string | null {
  const m = /HP[:_]?(\d{7})/i.exec(value) ?? /^\s*(\d{7})\s*$/.exec(value)
  return m ? `HP:${m[1]}` : null
}

/** Phenotypes as a term → weight map, whichever form they were given in. */
export function phenotypeWeights(input: DiseaseInput): Record<string, number> {
  const p = input.phenotypes
  if (!p) return {}
  return Array.isArray(p) ? Object.fromEntries(p.map((t) => [t, 0.5])) : p
}

export function newUserId(name: string): string {
  const slug = name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')
    .slice(0, 40)
  return `USER:${slug || 'disease'}-${Math.random().toString(36).slice(2, 7)}`
}

export function countFeatures(input: DiseaseInput): number {
  return (
    Object.keys(phenotypeWeights(input)).length +
    (input.genes?.length ?? 0) +
    (input.drugs?.length ?? 0) +
    (input.inheritance?.length ?? 0) +
    (input.onset?.length ?? 0) +
    (input.ontology_parents?.length ?? 0) +
    (input.prevalence ? 1 : 0) +
    (input.description?.trim() ? 1 : 0) +
    (input.synonyms?.length ?? 0)
  )
}

/** Drop empty fields so the stored JSON matches what a researcher would write by hand. */
export function compactInput(input: DiseaseInput): DiseaseInput {
  const out: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(input)) {
    if (v === undefined || v === null || v === '') continue
    if (Array.isArray(v) && v.length === 0) continue
    if (typeof v === 'object' && !Array.isArray(v) && Object.keys(v).length === 0) continue
    out[k] = typeof v === 'string' ? v.trim() : v
  }
  return out as unknown as DiseaseInput
}

// --- JSON import ---------------------------------------------------------------

export interface ImportItem {
  input: DiseaseInput | null
  errors: string[]
  warnings: string[]
}

/** Accepts one v2 disease object, a list, or {"diseases": [...]}. v2 itself reports unknown values on placement. */
export function parseImport(text: string): { items: ImportItem[]; error?: string } {
  let data: unknown
  try {
    data = JSON.parse(text)
  } catch (e) {
    return { items: [], error: `Not valid JSON: ${(e as Error).message}` }
  }
  const list = Array.isArray(data)
    ? data
    : data && typeof data === 'object' && Array.isArray((data as { diseases?: unknown }).diseases)
      ? (data as { diseases: unknown[] }).diseases
      : [data]
  if (list.length === 0) return { items: [], error: 'The file contains no diseases.' }
  return { items: list.map(importOne) }
}

const LIST_FIELDS = ['synonyms', 'genes', 'drugs', 'inheritance', 'onset', 'ontology_parents'] as const

function importOne(item: unknown): ImportItem {
  const errors: string[] = []
  const warnings: string[] = []
  if (!item || typeof item !== 'object' || Array.isArray(item)) {
    return { input: null, errors: ['Each disease must be a JSON object.'], warnings }
  }
  const raw = item as Record<string, unknown>
  const name = typeof raw.name === 'string' ? raw.name.trim() : ''
  if (!name) errors.push('Missing "name".')
  for (const key of Object.keys(raw)) if (!V2_FIELDS.has(key)) warnings.push(`Unknown field "${key}" will be ignored.`)

  const input: Record<string, unknown> = { ...raw, name }
  delete input.id
  if (raw.hpo_ids && !raw.phenotypes) {
    input.phenotypes = raw.hpo_ids
    delete input.hpo_ids
  }
  if (raw.parents && !raw.ontology_parents) {
    input.ontology_parents = raw.parents
    delete input.parents
  }
  const phenotypes = input.phenotypes
  if (phenotypes !== undefined && typeof phenotypes !== 'object') errors.push('"phenotypes" must be a list or an object of HPO term → weight.')
  for (const key of LIST_FIELDS) {
    if (input[key] !== undefined && !Array.isArray(input[key])) errors.push(`"${key}" must be a list.`)
  }
  if (input.prevalence !== undefined && !(typeof input.prevalence === 'number' && input.prevalence > 0)) {
    warnings.push('"prevalence" should be a positive fraction such as 1e-5; v2 will ignore it otherwise.')
  }
  return { input: errors.length ? null : (input as unknown as DiseaseInput), errors, warnings }
}

export const JSON_TEMPLATE = {
  name: 'Example: infantile epileptic encephalopathy (replace me)',
  synonyms: [],
  description: 'Free-text clinical description. v2 compares it with Orphanet descriptions.',
  phenotypes: { 'HP:0001250': 1.0, 'HP:0001263': 0.8 },
  genes: ['CDKL5'],
  drugs: [],
  inheritance: ['X-linked dominant'],
  onset: ['Infancy'],
  prevalence: 1e-6,
  ontology_parents: [],
}

export function exportJson(diseases: UserDisease[]): string {
  return JSON.stringify({ diseases: diseases.map((d) => d.input) }, null, 2)
}

// --- Persistence ---------------------------------------------------------------

const STORAGE_KEY = 'rare-disease-explorer.v2-user-diseases'

function load(): UserDisease[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    const parsed = raw ? JSON.parse(raw) : []
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

function save(diseases: UserDisease[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(diseases))
  } catch {
    // Storage may be unavailable (private mode, blocked site data); the
    // diseases still work for this session.
  }
}

/** Added diseases, kept in this browser's local storage. */
export function useUserDiseases() {
  const [diseases, setDiseases] = useState<UserDisease[]>(load)
  const commit = useCallback((update: (current: UserDisease[]) => UserDisease[]) => {
    setDiseases((current) => {
      const next = update(current)
      save(next)
      return next
    })
  }, [])
  const add = useCallback(
    (items: { input: DiseaseInput; labels?: Record<string, string> }[]): UserDisease[] => {
      const created = items.map((d) => ({
        id: newUserId(d.input.name),
        input: compactInput(d.input),
        labels: d.labels,
        createdAt: new Date().toISOString(),
      }))
      commit((current) => [...current, ...created])
      return created
    },
    [commit],
  )
  const update = useCallback(
    (id: string, change: Partial<Omit<UserDisease, 'id'>>) =>
      commit((current) => current.map((d) => (d.id === id ? { ...d, ...change } : d))),
    [commit],
  )
  const remove = useCallback((id: string) => commit((current) => current.filter((d) => d.id !== id)), [commit])
  return { diseases, add, update, remove }
}
