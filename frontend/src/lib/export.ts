import type { Disease } from '../data/types'
import type { EdgeView } from './graph'

/** Download the given pairs as CSV. */
export function exportCsv(edges: EdgeView[], diseases: Map<string, Disease>, measure: string) {
  const header = [
    'disease_a',
    'disease_a_orpha_id',
    'disease_b',
    'disease_b_orpha_id',
    'relationship',
    'overall_similarity',
    `${measure}_score`,
    'n_papers',
    'evidence_score',
    'best_study_design',
    'caveats',
    'pmids',
  ]
  const quote = (v: unknown) => {
    const s = v === null || v === undefined ? '' : String(v)
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
  }
  const lines = [header.join(',')]
  for (const { edge, agg, value, papers } of edges) {
    const a = diseases.get(edge.source)
    const b = diseases.get(edge.target)
    lines.push(
      [
        a?.name ?? edge.source,
        a?.orphaId ?? '',
        b?.name ?? edge.target,
        b?.orphaId ?? '',
        agg.relationship,
        agg.similarity,
        value,
        agg.nPapers,
        agg.evidenceScore,
        agg.bestDesign,
        agg.flags.join('; '),
        [...new Set(papers.map((p) => p.pmid).filter(Boolean))].join(' '),
      ]
        .map(quote)
        .join(','),
    )
  }
  const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = 'rare-disease-pairs.csv'
  link.click()
  URL.revokeObjectURL(url)
}
