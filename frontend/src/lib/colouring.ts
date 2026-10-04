// What the nodes are coloured by.  Cluster and category colours come straight from the graph file; the
// gene, symptom, onset, inheritance and similarity modes need the per-disease annotations, which are
// fetched the first time one of them is chosen.

import { useCallback, useEffect, useMemo, useState } from 'react'
import { type LoadedAnnotations, loadAnnotations } from '../data/annotations'
import type { ColourBy, DiseaseInput, GraphEdge, GraphNode } from '../data/types'
import { phenotypeWeights } from './userDiseases'
import { accentColour, type CanvasColours, categoricalColour, hslToHex, mix } from './theme'

export const COLOUR_MODES: { value: ColourBy; label: string; hint: string }[] = [
  { value: 'cluster', label: 'Cluster', hint: 'Louvain communities of the similarity graph' },
  { value: 'category', label: 'Orphanet category', hint: 'Top-level Orphanet classification' },
  { value: 'gene', label: 'Gene', hint: 'Diseases linked to one gene' },
  { value: 'symptom', label: 'Symptom (HPO)', hint: 'Diseases with one phenotype, or a subtype of it' },
  { value: 'onset', label: 'Age of onset', hint: 'Diseases that start at the same stage of life' },
  { value: 'inheritance', label: 'Inheritance', hint: 'Mode of inheritance' },
  { value: 'disease', label: 'Similarity to a disease', hint: 'A disease (ORPHA ID) and its most similar neighbours' },
  { value: 'plain', label: 'None', hint: 'One colour' },
]

/** Stages of life in order, so the colours run as a ramp. */
export const ONSET_ORDER = ['Antenatal', 'Neonatal', 'Infancy', 'Childhood', 'Adolescent', 'Adult', 'Elderly', 'All ages']
const MULTIPLE = 'Multiple modes'

export interface ColourGroup {
  key: string
  count: number
  colour: string
}

export interface NodeColouring {
  mode: ColourBy
  setMode: (mode: ColourBy) => void
  /** Gene symbol, HPO ID or ORPHA ID, depending on the mode. */
  term: string | null
  setTerm: (term: string | null) => void
  /** The disease similarity mode follows the selection while no disease is picked. */
  followsSelection: boolean
  /** Onset class / inheritance mode kept at full strength (the rest are dimmed). */
  groupFocus: string | null
  setGroupFocus: (key: string | null) => void
  colourOf: (d: GraphNode) => string
  /** Diseases carrying the chosen term or group; null when the mode highlights nothing in particular. */
  members: Set<string> | null
  groups: ColourGroup[]
  annotations: LoadedAnnotations | null
  loading: boolean
  error: string | null
  /** One-line description of what is coloured, e.g. "23 diseases linked to FBN1". */
  summary: string | null
}

interface Inputs {
  nodes: Map<string, GraphNode>
  userInputs: Map<string, DiseaseInput>
  edgesByNode: Map<string, GraphEdge[]>
  selectedDisease: string | null
  colours: CanvasColours
  defaultMode: ColourBy
  clusterColour: (index: number) => string
  categoryColour: (name: string) => string
}

const NEEDS_ANNOTATIONS: ColourBy[] = ['gene', 'symptom', 'onset', 'inheritance']

export function useNodeColouring(inputs: Inputs): NodeColouring {
  const { nodes, userInputs, edgesByNode, selectedDisease, colours, clusterColour, categoryColour } = inputs
  const [mode, setModeState] = useState<ColourBy>(inputs.defaultMode)
  const [term, setTerm] = useState<string | null>(null)
  const [groupFocus, setGroupFocus] = useState<string | null>(null)
  const [annotations, setAnnotations] = useState<LoadedAnnotations | null>(null)
  const [error, setError] = useState<string | null>(null)

  const setMode = useCallback((next: ColourBy) => {
    setModeState(next)
    setTerm(null)
    setGroupFocus(null)
  }, [])

  useEffect(() => {
    if (!NEEDS_ANNOTATIONS.includes(mode) || annotations) return
    let cancelled = false
    loadAnnotations().then(
      (a) => !cancelled && setAnnotations(a),
      (e: Error) => !cancelled && setError(e.message),
    )
    return () => {
      cancelled = true
    }
  }, [mode, annotations])
  const loading = NEEDS_ANNOTATIONS.includes(mode) && !annotations && !error

  const accent = accentColour(colours.dark)
  const diseaseTerm = mode === 'disease' ? (term ?? selectedDisease) : null

  // Disease → strength in (0, 1] for the modes that colour by a chosen term.
  const strength = useMemo(() => {
    const out = new Map<string, number>()
    if (mode === 'gene' && term && annotations) {
      for (const i of annotations.raw.genes[term] ?? []) out.set(annotations.raw.nodes[i], 1)
      for (const [id, input] of userInputs) if (input.genes?.includes(term)) out.set(id, 1)
    } else if (mode === 'symptom' && term && annotations) {
      const p = annotations.raw.phenotypes[term]
      if (p) p.n.forEach((i, k) => out.set(annotations.raw.nodes[i], Math.max(0.15, p.f[k] / 10)))
      for (const [id, input] of userInputs) {
        const weight = phenotypeWeights(input)[term]
        if (weight !== undefined) out.set(id, Math.max(0.15, weight))
      }
    } else if (diseaseTerm && nodes.has(diseaseTerm)) {
      const ranked = (edgesByNode.get(diseaseTerm) ?? [])
        .map((e) => ({ id: e.source === diseaseTerm ? e.target : e.source, score: e.score }))
        .sort((a, b) => b.score - a.score)
      ranked.forEach((r, k) => out.set(r.id, 1 - (k / Math.max(ranked.length, 1)) * 0.65))
      out.set(diseaseTerm, 1)
    }
    return out
  }, [mode, term, annotations, userInputs, diseaseTerm, nodes, edgesByNode])

  // Onset / inheritance: disease → the groups it belongs to.
  const grouping = useMemo(() => {
    const keys = mode === 'onset' ? ONSET_ORDER : null
    if ((mode !== 'onset' && mode !== 'inheritance') || !annotations) return null
    const source = mode === 'onset' ? annotations.raw.onset : annotations.raw.inheritance
    const all = new Map<string, string[]>()
    for (const [key, indices] of Object.entries(source)) {
      for (const i of indices) {
        const id = annotations.raw.nodes[i]
        all.set(id, [...(all.get(id) ?? []), key])
      }
    }
    for (const [id, input] of userInputs) {
      const values = (mode === 'onset' ? input.onset : input.inheritance) ?? []
      if (values.length > 0) all.set(id, values)
    }
    // A group's size is every disease that lists it (what a click shows); the colour is the earliest listed.
    const counts = new Map<string, number>()
    const primary = new Map<string, string>()
    for (const [id, values] of all) {
      const key =
        mode === 'onset'
          ? [...values].sort((a, b) => ONSET_ORDER.indexOf(a) - ONSET_ORDER.indexOf(b))[0]
          : values.length > 1
            ? MULTIPLE
            : values[0]
      primary.set(id, key)
      for (const value of new Set(values)) counts.set(value, (counts.get(value) ?? 0) + 1)
      if (mode === 'inheritance' && values.length > 1) counts.set(MULTIPLE, (counts.get(MULTIPLE) ?? 0) + 1)
    }
    const ordered = keys
      ? keys.filter((k) => counts.has(k))
      : [...counts.keys()].sort((a, b) => (counts.get(b) ?? 0) - (counts.get(a) ?? 0))
    const colourFor = (key: string) => {
      if (mode === 'onset') {
        if (key === 'All ages') return colours.dark ? '#9aa5b1' : '#6b7785'
        const i = ONSET_ORDER.indexOf(key)
        // Violet (before birth) through red (elderly).
        return hslToHex(290 - i * 46, 0.72, colours.dark ? 0.62 : 0.48)
      }
      return categoricalColour(ordered.indexOf(key), colours.dark)
    }
    const groups: ColourGroup[] = ordered.map((key) => ({ key, count: counts.get(key) ?? 0, colour: colourFor(key) }))
    return { all, primary, groups, colourOf: new Map(groups.map((g) => [g.key, g.colour])) }
  }, [mode, annotations, userInputs, colours.dark])

  const members = useMemo<Set<string> | null>(() => {
    if ((mode === 'gene' || mode === 'symptom' || mode === 'disease') && strength.size > 0) return new Set(strength.keys())
    if (grouping && groupFocus) {
      const out = new Set<string>()
      for (const [id, values] of grouping.all) {
        const hit = groupFocus === MULTIPLE ? values.length > 1 : values.includes(groupFocus)
        if (hit) out.add(id)
      }
      return out
    }
    return null
  }, [mode, strength, grouping, groupFocus])

  const colourOf = useCallback(
    (d: GraphNode): string => {
      switch (mode) {
        case 'cluster':
          return d.cluster >= 0 ? clusterColour(d.cluster) : colours.node
        case 'category':
          return d.origin !== 'user' && d.category ? categoryColour(d.category) : colours.node
        case 'gene':
        case 'symptom':
        case 'disease': {
          const s = strength.get(d.id)
          if (s === undefined) return colours.node
          if (mode === 'disease' && d.id === diseaseTerm) return colours.ink
          return mix(colours.node, accent, mode === 'gene' ? 1 : 0.3 + 0.7 * s)
        }
        case 'onset':
        case 'inheritance': {
          const key = grouping?.primary.get(d.id)
          return (key && grouping?.colourOf.get(key)) || colours.node
        }
        default:
          return colours.node
      }
    },
    [mode, clusterColour, categoryColour, colours, strength, accent, diseaseTerm, grouping],
  )

  const summary = useMemo(() => {
    if (mode === 'gene' && term) return `${strength.size.toLocaleString()} diseases linked to ${term}`
    if (mode === 'symptom' && term && annotations) {
      return `${strength.size.toLocaleString()} diseases with “${annotations.raw.phenotypes[term]?.l ?? term}”. Darker = more patients affected.`
    }
    if (mode === 'disease' && diseaseTerm && nodes.has(diseaseTerm)) {
      return `${nodes.get(diseaseTerm)!.name} and its ${(strength.size - 1).toLocaleString()} most similar diseases. Brighter = more similar.`
    }
    return null
  }, [mode, term, strength, annotations, diseaseTerm, nodes])

  return {
    mode,
    setMode,
    term,
    setTerm,
    followsSelection: mode === 'disease' && term === null,
    groupFocus,
    setGroupFocus,
    colourOf,
    members,
    groups: grouping?.groups ?? [],
    annotations,
    loading,
    error,
    summary,
  }
}
