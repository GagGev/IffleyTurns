import type { GraphEdge, GraphNode } from '../data/types'
import { download } from './format'

/** Download the given edges as CSV. */
export function exportCsv(edges: GraphEdge[], nodes: Map<string, GraphNode>) {
  const header = ['source', 'source_name', 'target', 'target_name', 'support', 'score', 'percentile', 'mutual', 'main_modality', 'origin']
  const quote = (v: unknown) => {
    const s = v === null || v === undefined ? '' : String(v)
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
  }
  const lines = [header.join(',')]
  for (const e of edges) {
    lines.push(
      [
        e.source,
        nodes.get(e.source)?.name ?? '',
        e.target,
        nodes.get(e.target)?.name ?? '',
        e.support,
        e.score,
        e.percentile,
        e.mutual,
        e.mainModality ?? '',
        e.origin,
      ]
        .map(quote)
        .join(','),
    )
  }
  download('rare-disease-similarity-edges.csv', lines.join('\n'), 'text/csv;charset=utf-8')
}
