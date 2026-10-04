import { useMemo, useState } from 'react'
import type { GraphEdge, GraphNode } from '../data/types'
import { capitalise, displayName, modalityLabel, score } from '../lib/format'
import { supportRank } from '../data/source'

interface Props {
  edges: GraphEdge[]
  nodes: Map<string, GraphNode>
  selectedEdgeId: string | null
  onSelectPair: (a: string, b: string) => void
}

type SortKey = 'a' | 'b' | 'support' | 'score' | 'main'

const MAX_ROWS = 500

export function EdgeTable({ edges, nodes, selectedEdgeId, onSelectPair }: Props) {
  const [sort, setSort] = useState<{ key: SortKey; desc: boolean }>({ key: 'score', desc: true })
  const name = (id: string) => displayName(nodes.get(id)?.name ?? id)

  const rows = useMemo(() => {
    const label = (id: string) => (nodes.get(id)?.name ?? id).toLowerCase()
    const keyOf = (e: GraphEdge): string | number => {
      switch (sort.key) {
        case 'a':
          return label(e.source)
        case 'b':
          return label(e.target)
        case 'support':
          return supportRank(e.support)
        case 'score':
          return e.score
        case 'main':
          return modalityLabel(e.mainModality)
      }
    }
    return [...edges].sort((x, y) => {
      const kx = keyOf(x)
      const ky = keyOf(y)
      const c = kx < ky ? -1 : kx > ky ? 1 : x.score - y.score
      return sort.desc ? -c : c
    })
  }, [edges, sort, nodes])

  const header = (key: SortKey, text: string, numeric = false) => (
    <th
      scope="col"
      className={numeric ? 'num' : undefined}
      aria-sort={sort.key === key ? (sort.desc ? 'descending' : 'ascending') : 'none'}
    >
      <button type="button" onClick={() => setSort((s) => ({ key, desc: s.key === key ? !s.desc : numeric }))}>
        {text}
        <span className="sort-mark" aria-hidden>
          {sort.key === key ? (sort.desc ? '↓' : '↑') : ''}
        </span>
      </button>
    </th>
  )

  return (
    <div className="table-wrap">
      <table className="edge-table">
        <thead>
          <tr>
            {header('a', 'Disease A')}
            {header('b', 'Disease B')}
            {header('support', 'Support')}
            {header('score', 'Score', true)}
            {header('main', 'Main evidence')}
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, MAX_ROWS).map((e) => (
            <tr
              key={e.id}
              className={e.id === selectedEdgeId ? 'is-selected' : undefined}
              tabIndex={0}
              onClick={() => onSelectPair(e.source, e.target)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                  event.preventDefault()
                  onSelectPair(e.source, e.target)
                }
              }}
            >
              <td>{name(e.source)}</td>
              <td>{name(e.target)}</td>
              <td className="nowrap">
                <span className={`line-key support-${e.support}${e.origin === 'user' ? ' is-user' : ''}`} aria-hidden />
                {capitalise(e.support)}
                {e.mutual && <span className="muted small"> · mutual</span>}
              </td>
              <td className="num">{score(e.score)}</td>
              <td>{modalityLabel(e.mainModality)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length > MAX_ROWS && (
        <p className="muted small table-note">
          Showing the {MAX_ROWS} highest of {rows.length.toLocaleString()} edges by the current sort. Narrow the filters
          or export to CSV for the rest.
        </p>
      )}
    </div>
  )
}
