import type { Measure } from './graph'
import { DIMENSION_LABELS } from '../data/types'

/** Capitalise names that the review recorded in lower case, leaving gene-style names alone. */
export function displayName(name: string): string {
  const first = name.split(/\s/)[0]
  return /^[a-z]+$/.test(first) ? name[0].toUpperCase() + name.slice(1) : name
}

export const score = (value: number | null | undefined) =>
  value === null || value === undefined ? '–' : value.toFixed(2)

export const measureLabel = (measure: Measure) =>
  measure === 'overall' ? 'Overall similarity' : `${DIMENSION_LABELS[measure]} similarity`

export function orphanetUrl(orphaId: string): string {
  return `https://www.orpha.net/en/disease/detail/${orphaId.replace(/^ORPHA:/, '')}`
}

export function pubmedUrl(pmid: string): string {
  return `https://pubmed.ncbi.nlm.nih.gov/${pmid}/`
}

export const plural = (n: number, word: string, many = `${word}s`) =>
  `${n.toLocaleString()} ${n === 1 ? word : many}`

export function capitalise(value: string): string {
  return value ? value[0].toUpperCase() + value.slice(1) : value
}
