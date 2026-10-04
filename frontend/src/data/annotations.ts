// Per-disease annotations and the literature pairs, loaded the first time they are needed.

import type { Annotations, PaperClaim, PaperEntry } from './types'

const dataUrl = (file: string) => `${import.meta.env.BASE_URL}data/${file}`

export interface TermOption {
  id: string
  label: string
  count: number
}

export interface LoadedAnnotations {
  raw: Annotations
  /** Disease ID → index into the arrays of `raw`. */
  index: Map<string, number>
  genes: TermOption[]
  symptoms: TermOption[]
}

let annotationsPromise: Promise<LoadedAnnotations> | null = null

export function loadAnnotations(): Promise<LoadedAnnotations> {
  if (!annotationsPromise) {
    annotationsPromise = fetch(dataUrl('annotations.json')).then(async (response) => {
      if (!response.ok || !response.headers.get('content-type')?.includes('json')) {
        throw new Error('annotations.json not found. Run python3 frontend/scripts/build_annotations.py')
      }
      const raw: Annotations = await response.json()
      return {
        raw,
        index: new Map(raw.nodes.map((id, i) => [id, i])),
        genes: Object.entries(raw.genes)
          .map(([id, n]) => ({ id, label: id, count: n.length }))
          .sort((a, b) => b.count - a.count),
        symptoms: Object.entries(raw.phenotypes)
          .map(([id, p]) => ({ id, label: p.l, count: p.n.length }))
          .sort((a, b) => b.count - a.count),
      }
    })
    annotationsPromise.catch(() => (annotationsPromise = null))
  }
  return annotationsPromise
}

/** Matches whose label or ID starts with the query first, then those that contain it; ties by prevalence. */
export function searchTerms(options: TermOption[], query: string, limit = 8): TermOption[] {
  const q = query.trim().toLowerCase()
  if (q.length === 0) return options.slice(0, limit)
  const exact: TermOption[] = []
  const starts: TermOption[] = []
  const contains: TermOption[] = []
  for (const o of options) {
    const label = o.label.toLowerCase()
    const id = o.id.toLowerCase()
    if (label === q || id === q) exact.push(o)
    else if (label.startsWith(q) || id.startsWith(q)) starts.push(o)
    else if (contains.length < limit && (label.includes(q) || id.includes(q))) contains.push(o)
  }
  // Among prefix matches, shorter names are closer to what was typed; options arrive ordered by prevalence.
  starts.sort((a, b) => a.label.length - b.label.length)
  return [...exact, ...starts, ...contains].slice(0, limit)
}

interface RawLiterature {
  papers: {
    id: string
    title: string
    year: number | null
    link: string
    studyType: string
    claims: PaperClaim[]
  }[]
  /** Paper dimension → v2 modalities that would carry it. */
  dimensions: Record<string, string[]>
  /** The latest acquisition release, minus papers already above; claims are unreviewed co-mentions. */
  acquired?: { id: string; title: string; year: number | null; link: string; access: string; claims: PaperClaim[] }[]
  acquiredRelease?: string | null
}

export interface Literature {
  papers: PaperEntry[]
  dimensions: Record<string, string[]>
  acquired: PaperEntry[]
  /** Date of the acquisition release, e.g. 2026-10-04. */
  acquiredRelease: string | null
}

let literaturePromise: Promise<Literature> | null = null

export function loadLiterature(): Promise<Literature> {
  if (!literaturePromise) {
    literaturePromise = fetch(dataUrl('literature.json')).then(async (response) => {
      if (!response.ok || !response.headers.get('content-type')?.includes('json')) {
        throw new Error('literature.json not found. Run python3 frontend/scripts/build_annotations.py')
      }
      const raw: RawLiterature = await response.json()
      return {
        dimensions: raw.dimensions,
        papers: raw.papers.map((p) => ({
          id: p.id,
          title: p.title,
          year: p.year,
          link: p.link || undefined,
          kind: 'literature' as const,
          mock: false,
          claims: p.claims,
        })),
        acquired: (raw.acquired ?? []).map((p) => ({
          id: p.id,
          title: p.title,
          year: p.year,
          link: p.link || undefined,
          kind: 'acquired' as const,
          mock: false,
          access: p.access,
          claims: p.claims,
        })),
        acquiredRelease: raw.acquiredRelease ?? null,
      }
    })
    literaturePromise.catch(() => (literaturePromise = null))
  }
  return literaturePromise
}
