// Diseases a researcher adds in the browser, either through the form or by
// uploading JSON.  The JSON format uses the diseases.parquet column names, so a
// row exported from the backend can be uploaded unchanged.

import { useCallback, useState } from 'react'
import {
  FAMILY_COLUMNS,
  type FeatureCatalogue,
  type FeatureFamily,
  type FeatureProfile,
  SET_FAMILIES,
} from '../data/features'

export type UserStatus = 'rare' | 'common' | 'unknown'

export interface UserDisease {
  id: string
  name: string
  orpha_id?: string
  status: UserStatus
  notes?: string
  hpo_ids: string[]
  gene_symbols: string[]
  category_ids: string[]
  body_system_ids: string[]
  inheritance: string[]
  onset: string[]
  approved_drug_ids: string[]
  prevalence_class?: string
  prevalence_estimated_per_person?: number
  createdAt: string
}

export type FamilyColumn = (typeof FAMILY_COLUMNS)[FeatureFamily]
const COLUMN_FAMILY = Object.fromEntries(SET_FAMILIES.map((f) => [FAMILY_COLUMNS[f], f])) as Record<string, FeatureFamily>

// Orphanet vocabularies, used when no exported catalogue supplies its own.
export const INHERITANCE_OPTIONS = [
  'Autosomal dominant',
  'Autosomal recessive',
  'X-linked recessive',
  'X-linked dominant',
  'Mitochondrial inheritance',
  'Multigenic/multifactorial',
  'Semi-dominant',
  'Y-linked',
  'Oligogenic',
  'Not applicable',
  'Unknown',
]
export const ONSET_OPTIONS = ['Antenatal', 'Neonatal', 'Infancy', 'Childhood', 'Adolescent', 'Adult', 'Elderly', 'All ages']
export const PREVALENCE_CLASSES = [
  '<1 / 1 000 000',
  '1-9 / 1 000 000',
  '1-9 / 100 000',
  '1-5 / 10 000',
  '6-9 / 10 000',
  '>1 / 1000',
]

/** Port of generate_features.prevalence_class_estimate. */
export function prevalenceClassEstimate(value: string): number | undefined {
  const s = value.replace(/[, ]/g, '')
  if (!s || s.toLowerCase() === 'unknown') return undefined
  let m = /^(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\/(\d+)$/.exec(s)
  if (m) return (Number(m[1]) + Number(m[2])) / 2 / Number(m[3])
  m = /^([<>])(\d+(?:\.\d+)?)\/(\d+)$/.exec(s)
  if (m) {
    const estimate = Number(m[2]) / Number(m[3])
    return m[1] === '<' ? estimate / 2 : estimate
  }
  m = /^(\d+(?:\.\d+)?)\/(\d+)$/.exec(s)
  if (m) return Number(m[1]) / Number(m[2])
  return undefined
}

// --- Term normalisation -------------------------------------------------------

/** Normalise `58`, `ORPHA_58`, `Orphanet:58` to `ORPHA:58`, as evaluation.normalize_orpha_id does. */
export function normaliseOrpha(value: string): string | null {
  const m = /^\s*(?:(?:ORPHA|ORPHANET)\s*[:_-]?\s*)?(\d+)\s*$/i.exec(value)
  return m ? `ORPHA:${Number(m[1])}` : null
}

export function normaliseHpo(value: string): string | null {
  const m = /HP[:_]?(\d{7})/i.exec(value) ?? /^\s*(\d{7})\s*$/.exec(value)
  return m ? `HP:${m[1]}` : null
}

export interface TermCheck {
  value: string | null
  /** Why the token was rejected, or a hint about how it will be matched. */
  message?: string
  known: boolean
}

/**
 * Resolve one user-entered token for a family: map names and differently-cased
 * IDs onto the catalogue's term IDs where possible, and validate ID formats.
 */
export function resolveTerm(family: FeatureFamily, raw: string, catalogue: FeatureCatalogue | null): TermCheck {
  const token = raw.trim()
  if (!token) return { value: null, known: false, message: 'Empty value' }
  const lookup = catalogue ? termLookup(catalogue, family) : null

  if (family === 'phenotypes') {
    const id = normaliseHpo(token)
    if (id) return { value: id, known: !lookup || lookup.has(id.toLowerCase()) }
    const byName = lookup?.get(token.toLowerCase())
    if (byName) return { value: byName, known: true }
    return { value: null, known: false, message: 'Not an HPO ID (e.g. HP:0001166)' }
  }

  const exact = lookup?.get(token.toLowerCase())
  if (exact) return { value: exact, known: true }
  if (family === 'classifications' || family === 'body_systems') {
    const orpha = normaliseOrpha(token)
    if (orpha) return { value: orpha, known: !lookup || lookup.has(orpha.toLowerCase()) }
    if (/^[A-Z]+[_:][A-Za-z0-9]+$/.test(token)) return { value: token, known: !lookup }
    return { value: null, known: false, message: 'Use an ORPHA or EFO ID, or pick a suggestion' }
  }
  return { value: token, known: !lookup }
}

const lookupCache = new WeakMap<FeatureCatalogue, Map<FeatureFamily, Map<string, string>>>()

/** Lower-cased term ID or name → canonical term ID. */
function termLookup(catalogue: FeatureCatalogue, family: FeatureFamily): Map<string, string> {
  let perFamily = lookupCache.get(catalogue)
  if (!perFamily) {
    perFamily = new Map()
    lookupCache.set(catalogue, perFamily)
  }
  let map = perFamily.get(family)
  if (!map) {
    map = new Map()
    for (const profile of catalogue.profiles) {
      for (const term of profile.sets[family] ?? []) map.set(term.toLowerCase(), term)
    }
    for (const [term, name] of catalogue.termNames[family]) {
      if (!map.has(name.toLowerCase())) map.set(name.toLowerCase(), term)
    }
    perFamily.set(family, map)
  }
  return map
}

// --- Conversion ----------------------------------------------------------------

export function toProfile(disease: UserDisease, catalogue: FeatureCatalogue | null): FeatureProfile {
  const sets: FeatureProfile['sets'] = {}
  for (const family of SET_FAMILIES) {
    const values = new Set<string>()
    for (const raw of disease[FAMILY_COLUMNS[family] as FamilyColumn] as string[]) {
      const { value } = resolveTerm(family, raw, catalogue)
      if (value) values.add(value)
    }
    if (values.size) sets[family] = values
  }
  const prevalence =
    disease.prevalence_estimated_per_person ??
    (disease.prevalence_class ? prevalenceClassEstimate(disease.prevalence_class) : undefined)
  return { id: disease.id, name: disease.name, sets, prevalence, prevalenceClass: disease.prevalence_class }
}

export function newUserId(name: string): string {
  const slug = name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')
    .slice(0, 40)
  return `user:${slug || 'disease'}-${Math.random().toString(36).slice(2, 7)}`
}

export function emptyUserDisease(): UserDisease {
  return {
    id: '',
    name: '',
    status: 'unknown',
    hpo_ids: [],
    gene_symbols: [],
    category_ids: [],
    body_system_ids: [],
    inheritance: [],
    onset: [],
    approved_drug_ids: [],
    createdAt: '',
  }
}

// --- JSON import ---------------------------------------------------------------

const ALIASES: Record<string, string> = {
  phenotypes: 'hpo_ids',
  hpo: 'hpo_ids',
  genes: 'gene_symbols',
  categories: 'category_ids',
  classifications: 'category_ids',
  body_systems: 'body_system_ids',
  drugs: 'approved_drug_ids',
  approved_drugs: 'approved_drug_ids',
  prevalence: 'prevalence_class',
  prevalence_per_person: 'prevalence_estimated_per_person',
  orpha: 'orpha_id',
  orphaId: 'orpha_id',
}
const IGNORED = new Set(['id', 'createdAt', 'body_system_names', 'category_names', 'drug_names', 'synonyms', 'description'])

export interface ImportItem {
  disease: UserDisease | null
  errors: string[]
  warnings: string[]
}

export function parseImport(text: string, catalogue: FeatureCatalogue | null): { items: ImportItem[]; error?: string } {
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
  return { items: list.map((item) => importOne(item, catalogue)) }
}

function importOne(item: unknown, catalogue: FeatureCatalogue | null): ImportItem {
  const errors: string[] = []
  const warnings: string[] = []
  if (!item || typeof item !== 'object' || Array.isArray(item)) {
    return { disease: null, errors: ['Each disease must be a JSON object.'], warnings }
  }
  const d = emptyUserDisease()
  for (const [rawKey, value] of Object.entries(item as Record<string, unknown>)) {
    const key = ALIASES[rawKey] ?? rawKey
    if (value === null || value === undefined || value === '') continue
    if (key === 'name') {
      d.name = String(value).trim()
    } else if (key === 'orpha_id') {
      const orpha = normaliseOrpha(String(value))
      if (orpha) d.orpha_id = orpha
      else warnings.push(`Ignored orpha_id "${value}": not an ORPHA ID.`)
    } else if (key === 'status') {
      const s = String(value).toLowerCase()
      d.status = s === 'rare' || s === 'common' ? s : 'unknown'
    } else if (key === 'notes') {
      d.notes = String(value)
    } else if (key === 'prevalence_class') {
      d.prevalence_class = String(value)
      if (prevalenceClassEstimate(d.prevalence_class) === undefined) {
        warnings.push(`Prevalence "${value}" is not an Orphanet class such as "1-9 / 100 000"; it will not be compared.`)
      }
    } else if (key === 'prevalence_estimated_per_person') {
      const n = Number(value)
      if (Number.isFinite(n) && n > 0 && n < 1) d.prevalence_estimated_per_person = n
      else warnings.push(`Ignored prevalence_estimated_per_person ${JSON.stringify(value)}: must be between 0 and 1.`)
    } else if (key in COLUMN_FAMILY) {
      const family = COLUMN_FAMILY[key]
      const values = Array.isArray(value) ? value : String(value).split(/[,;\n]/)
      const kept: string[] = []
      const rejected: string[] = []
      const unknown: string[] = []
      for (const v of values) {
        const check = resolveTerm(family, String(v), catalogue)
        if (!check.value) rejected.push(String(v))
        else {
          kept.push(check.value)
          if (!check.known) unknown.push(check.value)
        }
      }
      ;(d[key as FamilyColumn] as string[]) = [...new Set(kept)]
      if (rejected.length) warnings.push(`Dropped from ${key}: ${rejected.slice(0, 5).join(', ')}${rejected.length > 5 ? '…' : ''}`)
      if (unknown.length && catalogue) {
        warnings.push(`${unknown.length} ${key} value(s) appear in no catalogue disease, so they cannot match anything.`)
      }
    } else if (!IGNORED.has(rawKey)) {
      warnings.push(`Unknown field "${rawKey}" ignored.`)
    }
  }
  if (!d.name) errors.push('Missing "name".')
  return { disease: errors.length ? null : d, errors, warnings }
}

export const JSON_TEMPLATE = {
  diseases: [
    {
      name: 'Example disease (replace me)',
      orpha_id: null,
      status: 'rare',
      hpo_ids: ['HP:0001166', 'HP:0001519'],
      gene_symbols: ['FBN1'],
      inheritance: ['Autosomal dominant'],
      onset: ['Childhood'],
      category_ids: [],
      body_system_ids: [],
      approved_drug_ids: [],
      prevalence_class: '1-5 / 10 000',
      notes: 'Every field except name is optional. Leave unknown fields out or empty.',
    },
  ],
}

export function exportJson(diseases: UserDisease[]): string {
  return JSON.stringify(
    {
      diseases: diseases.map((d) => {
        const copy: Partial<UserDisease> = { ...d }
        delete copy.id
        delete copy.createdAt
        return copy
      }),
    },
    null,
    2,
  )
}

// --- Persistence ---------------------------------------------------------------

const STORAGE_KEY = 'rare-disease-explorer.user-diseases.v1'

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
    // Storage can be unavailable (private mode, blocked site data); the
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
    (items: UserDisease[]): UserDisease[] => {
      const created = items.map((d) => ({ ...d, id: newUserId(d.name), createdAt: new Date().toISOString() }))
      commit((current) => [...current, ...created])
      return created
    },
    [commit],
  )
  const update = useCallback(
    (disease: UserDisease) => commit((current) => current.map((d) => (d.id === disease.id ? disease : d))),
    [commit],
  )
  const remove = useCallback((id: string) => commit((current) => current.filter((d) => d.id !== id)), [commit])
  return { diseases, add, update, remove }
}
