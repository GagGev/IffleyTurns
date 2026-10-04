import { useEffect, useMemo, useState } from 'react'
import { expandDetail, loadEdgeDetail } from '../data/source'
import type { EdgeDetail, GraphData, GraphEdge, PaperClaim, PaperEntry } from '../data/types'
import { type ClaimEvaluation, claimEdge, evaluateClaim } from './paperEval'

/** Judge every claim of one paper; graph-edge explanations not baked into the claim are fetched. */
export function usePaperEvaluation(
  paper: PaperEntry | null,
  graph: GraphData,
  edgesById: Map<string, GraphEdge>,
  userDetails: Map<string, EdgeDetail>,
): ClaimEvaluation[] {
  const [fetched, setFetched] = useState<Map<string, EdgeDetail | null>>(new Map())

  const needed = useMemo(() => {
    if (!paper) return []
    const ids = new Set<string>()
    for (const claim of paper.claims) {
      const edge = claimEdge(edgesById, claim)
      if (edge && edge.origin === 'graph' && !claim.detail) ids.add(edge.id)
    }
    return [...ids]
  }, [paper, edgesById])

  useEffect(() => {
    const missing = needed.filter((id) => !fetched.has(id))
    if (missing.length === 0) return
    let cancelled = false
    Promise.all(missing.map((id) => loadEdgeDetail(graph, id).catch(() => null))).then((details) => {
      if (cancelled) return
      setFetched((current) => {
        const next = new Map(current)
        missing.forEach((id, i) => next.set(id, details[i]))
        return next
      })
    })
    return () => {
      cancelled = true
    }
  }, [needed, fetched, graph])

  return useMemo(() => {
    if (!paper) return []
    return paper.claims.map((claim) => evaluateClaim(claim, ...edgeAndDetail(claim, graph, edgesById, userDetails, fetched)))
  }, [paper, graph, edgesById, userDetails, fetched])
}

function edgeAndDetail(
  claim: PaperClaim,
  graph: GraphData,
  edgesById: Map<string, GraphEdge>,
  userDetails: Map<string, EdgeDetail>,
  fetched: Map<string, EdgeDetail | null>,
): [GraphEdge | null, EdgeDetail | null] {
  const edge = claimEdge(edgesById, claim)
  if (!edge) return [null, null]
  if (edge.origin === 'user') return [edge, userDetails.get(edge.id) ?? null]
  if (claim.detail) return [edge, expandDetail(graph.modalities, claim.detail)]
  // A graph edge whose explanation failed to load stays pending (null) rather than being judged on nothing.
  return [edge, fetched.get(edge.id) ?? null]
}

/** Judge a whole set of papers at once; only possible for claims with a baked-in explanation (the literature set). */
export function evaluateLiterature(papers: PaperEntry[], graph: GraphData, edgesById: Map<string, GraphEdge>) {
  const out = new Map<string, ClaimEvaluation[]>()
  for (const paper of papers) {
    out.set(
      paper.id,
      paper.claims.map((claim) => {
        const edge = claimEdge(edgesById, claim)
        const detail = edge && claim.detail ? expandDetail(graph.modalities, claim.detail) : null
        return evaluateClaim(claim, edge, detail)
      }),
    )
  }
  return out
}
