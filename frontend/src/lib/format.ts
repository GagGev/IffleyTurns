import { MODALITY_LABELS } from '../data/types'

/** Capitalise names recorded in lower case, leaving gene-style names alone. */
export function displayName(name: string): string {
  const first = name.split(/\s/)[0]
  return /^[a-z]+$/.test(first) ? name[0].toUpperCase() + name.slice(1) : name
}

export const score = (value: number | null | undefined, digits = 2) =>
  value === null || value === undefined ? '–' : value.toFixed(digits)

export const modalityLabel = (m: string | null | undefined) => (m ? (MODALITY_LABELS[m] ?? m) : '–')

export function orphanetUrl(orphaId: string): string {
  return `https://www.orpha.net/en/disease/detail/${orphaId.replace(/^ORPHA:/, '')}`
}

/** Cluster labels follow the build script: C1 is the largest cluster. */
export const clusterId = (index: number) => (index >= 0 ? `C${index + 1}` : '–')

export const isOrpha = (id: string) => /^ORPHA:\d+$/.test(id)

export const plural = (n: number, word: string, many = `${word}s`) =>
  `${n.toLocaleString()} ${n === 1 ? word : many}`

export function capitalise(value: string): string {
  return value ? value[0].toUpperCase() + value.slice(1) : value
}

export function download(name: string, text: string, type: string) {
  const url = URL.createObjectURL(new Blob([text], { type }))
  const a = document.createElement('a')
  a.href = url
  a.download = name
  a.click()
  URL.revokeObjectURL(url)
}
