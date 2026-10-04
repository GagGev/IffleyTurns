import { MODALITY_LABELS } from '../data/types'

/** Capitalise names recorded in lower case, leaving gene-style names alone. */
export function displayName(name: string): string {
  const first = name.split(/\s/)[0]
  return /^[a-z]+$/.test(first) ? name[0].toUpperCase() + name.slice(1) : name
}

export const score = (value: number | null | undefined, digits = 2) =>
  value === null || value === undefined ? '–' : value.toFixed(digits)

/** 0.99973 → "99.97%"; keeps enough digits to separate values near 100%. */
export function percentile(value: number): string {
  if (value >= 1) return '100%'
  const pct = value * 100
  const digits = pct >= 99.99 ? 3 : pct >= 99 ? 2 : pct >= 90 ? 1 : 0
  return `${pct.toFixed(digits)}%`
}

/** "Top 0.03% of random pairs". */
export function percentileSentence(value: number): string {
  const top = (1 - value) * 100
  if (top <= 0) return 'Scores above every sampled random pair'
  const shown = top < 0.01 ? '<0.01' : top < 1 ? top.toFixed(2) : top.toFixed(1)
  return `Top ${shown}% of random disease pairs`
}

export const modalityLabel = (m: string | null | undefined) => (m ? (MODALITY_LABELS[m] ?? m) : '–')

export function orphanetUrl(orphaId: string): string {
  return `https://www.orpha.net/en/disease/detail/${orphaId.replace(/^ORPHA:/, '')}`
}

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
