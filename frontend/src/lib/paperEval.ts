// Does the graph carry the evidence a paper says it should?
//
// A paper claim says: diseases A and B are similar, in these respects (phenotype, genes, pathways, ...).
// The graph answers with an edge, if A and B are among each other's most similar diseases, and for that
// edge a contribution from each of v2's modalities.  A claim is judged on two things:
//
//   1. is there an edge?                              no  -> missing
//   2. does it get score from the claimed modalities?  all -> confirmed, some -> partial, none -> unsupported
//
// Claims whose dimension has no v2 modality (diagnostic, comorbidity) can only be judged on the edge: edge-only.

import type { ClaimVerdict, EdgeDetail, GraphEdge, PaperClaim } from '../data/types'
import { findEdge } from './graph'

/** A modality "carries" a dimension when its contribution to the edge score is at least this. */
export const CARRY_MIN = 0.25

/** Paper dimension → v2 modalities that would express it (same table as scripts/build_annotations.py). */
export const DIMENSION_MODALITIES: Record<string, string[]> = {
  phenotype: ['phenotype'],
  genes: ['gene', 'ot_gene'],
  pathways: ['pathway'],
  treatment: ['drug', 'drug_target'],
  epidemiology: ['prevalence', 'onset', 'inheritance'],
  'other (diagnostic)': [],
  'other (comorbidity)': [],
}

export const VERDICT_ORDER: ClaimVerdict[] = ['confirmed', 'partial', 'unsupported', 'edge-only', 'missing']

export const VERDICT_LABEL: Record<ClaimVerdict, string> = {
  confirmed: 'Carried',
  partial: 'Partly carried',
  unsupported: 'Edge, other reasons',
  'edge-only': 'Edge present',
  missing: 'No edge',
}

export const VERDICT_MEANING: Record<ClaimVerdict, string> = {
  confirmed: 'The graph has this edge and every dimension the paper names contributes to its score.',
  partial: 'The graph has this edge, but only some of the dimensions the paper names contribute to its score.',
  unsupported: 'The graph has this edge, but none of the dimensions the paper names contribute: the right pair for other reasons.',
  'edge-only': 'The graph has this edge. The paper’s dimension (diagnostic, comorbidity) has no modality in v2, so nothing more can be checked.',
  missing: 'Neither disease lists the other among its most similar diseases: the graph does not carry this relation.',
}

export interface DimensionCheck {
  dimension: string
  /** Modalities that could carry it; empty when v2 has none. */
  modalities: string[]
  /** Each of those modalities' contribution to the edge, strongest first. */
  contributions: { modality: string; contribution: number; similarity: number | null }[]
  /** True/false once an edge exists; null when v2 has no modality for the dimension. */
  carried: boolean | null
  /** False when none of its modalities is annotated for both diseases, so the graph had nothing to compare. */
  annotated: boolean
}

export interface ClaimEvaluation {
  claim: PaperClaim
  edge: GraphEdge | null
  /** Explanation of the edge; null while loading, or when there is no edge. */
  detail: EdgeDetail | null
  dimensions: DimensionCheck[]
  /** Undefined while the edge's explanation is still loading. */
  verdict: ClaimVerdict | undefined
  /** The feature the paper names, found among the edge's shared features (null if none was named). */
  featureShared: boolean | null
}

export function judge(dimensions: DimensionCheck[], hasEdge: boolean): ClaimVerdict {
  if (!hasEdge) return 'missing'
  const checkable = dimensions.filter((d) => d.carried !== null)
  if (checkable.length === 0) return 'edge-only'
  const carried = checkable.filter((d) => d.carried).length
  return carried === checkable.length ? 'confirmed' : carried > 0 ? 'partial' : 'unsupported'
}

export function evaluateClaim(
  claim: PaperClaim,
  edge: GraphEdge | null,
  detail: EdgeDetail | null,
): ClaimEvaluation {
  const dimensions: DimensionCheck[] = claim.dimensions.map((dimension) => {
    const modalities = DIMENSION_MODALITIES[dimension] ?? []
    const contributions = modalities
      .map((modality) => ({
        modality,
        contribution: detail?.contributions[modality] ?? 0,
        similarity: detail?.similarities[modality] ?? null,
      }))
      .sort((a, b) => b.contribution - a.contribution)
    return {
      dimension,
      modalities,
      contributions,
      carried: modalities.length === 0 ? null : contributions.some((c) => c.contribution >= CARRY_MIN),
      annotated: contributions.some((c) => c.similarity !== null),
    }
  })
  const pending = edge !== null && detail === null
  const needle = claim.feature?.toLowerCase()
  const featureShared =
    needle && detail ? detail.evidence.some((e) => e.shared.some((s) => s.toLowerCase().includes(needle))) : null
  return {
    claim,
    edge,
    detail,
    dimensions,
    verdict: pending ? undefined : judge(dimensions, edge !== null),
    featureShared,
  }
}

export function countVerdicts(evaluations: ClaimEvaluation[]): Record<ClaimVerdict, number> {
  const counts: Record<ClaimVerdict, number> = { confirmed: 0, partial: 0, unsupported: 0, 'edge-only': 0, missing: 0 }
  for (const e of evaluations) if (e.verdict) counts[e.verdict]++
  return counts
}

/** Edge for a claim, looked up in either direction. */
export function claimEdge(edgesById: Map<string, GraphEdge>, claim: PaperClaim): GraphEdge | null {
  return findEdge(edgesById, claim.a, claim.b) ?? null
}

/** Share of judged claims in which the graph has the edge. */
export function recall(counts: Record<ClaimVerdict, number>): number {
  const total = VERDICT_ORDER.reduce((s, v) => s + counts[v], 0)
  return total === 0 ? 0 : (total - counts.missing) / total
}
