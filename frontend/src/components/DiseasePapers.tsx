import { useMemo, useState } from 'react'
import type { Literature } from '../data/annotations'
import type { GraphNode, PaperEntry } from '../data/types'
import { displayName, plural } from '../lib/format'
import { InfoTip } from './InfoTip'

interface PaperMention {
  paper: PaperEntry
  /** The other diseases this paper discusses alongside the selected one. */
  others: string[]
}

const SHOWN_COLLAPSED = 3

const indexes = new WeakMap<Literature, Map<string, PaperMention[]>>()

/** Disease ID -> papers whose claims involve it; curated papers first, then newest. */
function papersByDisease(literature: Literature): Map<string, PaperMention[]> {
  let index = indexes.get(literature)
  if (index) return index
  index = new Map()
  for (const paper of [...literature.papers, ...literature.acquired]) {
    const partners = new Map<string, Set<string>>()
    for (const claim of paper.claims) {
      for (const [self, other] of [
        [claim.a, claim.b],
        [claim.b, claim.a],
      ]) {
        if (!self) continue
        const set = partners.get(self) ?? new Set<string>()
        if (other) set.add(other)
        partners.set(self, set)
      }
    }
    for (const [disease, others] of partners) {
      const list = index.get(disease) ?? []
      list.push({ paper, others: [...others] })
      index.set(disease, list)
    }
  }
  for (const list of index.values()) {
    list.sort(
      (x, y) =>
        Number(y.paper.kind === 'literature') - Number(x.paper.kind === 'literature') ||
        (y.paper.year ?? 0) - (x.paper.year ?? 0),
    )
  }
  indexes.set(literature, index)
  return index
}

interface Props {
  id: string
  literature: Literature | null
  error: string | null
  nodes: Map<string, GraphNode>
  onSelectPair: (a: string, b: string) => void
  onOpenPaper: (paperId: string) => void
}

/** Papers in the project's literature set that discuss a disease, with a toggle to list them all. */
export function DiseasePapers({ id, literature, error, nodes, onSelectPair, onOpenPaper }: Props) {
  const [expanded, setExpanded] = useState(false)
  const mentions = useMemo(() => (literature ? (papersByDisease(literature).get(id) ?? []) : []), [literature, id])
  const shown = expanded ? mentions : mentions.slice(0, SHOWN_COLLAPSED)
  const name = (other: string) => displayName(nodes.get(other)?.name ?? other)

  return (
    <section className="panel-section">
      <h3 className="label-row">
        <span>
          Papers in the literature set <span className="muted">· {literature ? mentions.length : '…'}</span>
        </span>
        <InfoTip topic="diseasePapers" />
      </h3>
      {error ? (
        <p className="muted small">{error}</p>
      ) : !literature ? (
        <p className="muted small">Loading papers…</p>
      ) : mentions.length === 0 ? (
        <p className="muted small">No paper in the literature set discusses this disease.</p>
      ) : (
        <>
          <ul className="disease-papers">
            {shown.map(({ paper, others }) => (
              <li key={paper.id}>
                {paper.link ? (
                  <a href={paper.link} target="_blank" rel="noreferrer" className="disease-paper-title">
                    {paper.title}
                  </a>
                ) : (
                  <span className="disease-paper-title">{paper.title}</span>
                )}
                <span className="muted small">
                  {[paper.year, paper.kind === 'literature' ? 'Curated' : 'Unreviewed co-mention'].filter(Boolean).join(' · ')}
                </span>
                {others.length > 0 && (
                  <span className="small disease-paper-with">
                    With{' '}
                    {others.map((other, i) => (
                      <span key={other}>
                        {i > 0 && ', '}
                        <button type="button" className="link-button" onClick={() => onSelectPair(id, other)}>
                          {name(other)}
                        </button>
                      </span>
                    ))}
                  </span>
                )}
                <button type="button" className="link-button small" onClick={() => onOpenPaper(paper.id)}>
                  Show its claims on the graph
                </button>
              </li>
            ))}
          </ul>
          {mentions.length > SHOWN_COLLAPSED && (
            <button type="button" className="link-button small" aria-expanded={expanded} onClick={() => setExpanded(!expanded)}>
              {expanded ? 'Show fewer' : `Show all ${plural(mentions.length, 'paper')}`}
            </button>
          )}
        </>
      )}
    </section>
  )
}
