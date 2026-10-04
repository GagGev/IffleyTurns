import { useMemo, useState } from 'react'
import type { Disease } from '../data/types'
import type { EdgeView } from '../lib/graph'
import { capitalise, displayName, score } from '../lib/format'
import type { Selection } from '../lib/selection'

interface Props {
  edges: EdgeView[]
  diseases: Map<string, Disease>
  selection: Selection
  measureLabel: string
  onSelectPair: (a: string, b: string) => void
}

type SortKey = 'a' | 'b' | 'relationship' | 'value' | 'papers' | 'evidence'

const MAX_ROWS = 500

export function EdgeTable({ edges, diseases, selection, measureLabel, onSelectPair }: Props) {
  const [sort, setSort] = useState<{ key: SortKey; desc: boolean }>({ key: 'evidence', desc: true })
  const name = (id: string) => displayName(diseases.get(id)?.name ?? id)

  const rows = useMemo(() => {
    const name = (id: string) => displayName(diseases.get(id)?.name ?? id)
    const keyOf = (v: EdgeView): string | number => {
      switch (sort.key) {
        case 'a':
          return name(v.edge.source).toLowerCase()
        case 'b':
          return name(v.edge.target).toLowerCase()
        case 'relationship':
          return v.agg.relationship
        case 'value':
          return v.value
        case 'papers':
          return v.agg.nPapers
        case 'evidence':
          return v.agg.evidenceScore
      }
    }
    const sorted = [...edges].sort((x, y) => {
      const kx = keyOf(x)
      const ky = keyOf(y)
      const c = kx < ky ? -1 : kx > ky ? 1 : y.value - x.value
      return sort.desc ? -c : c
    })
    return sorted
  }, [edges, sort, diseases])

  const selectedPair = selection?.kind === 'pair' ? [selection.a, selection.b].sort().join('|') : null
  const header = (key: SortKey, label: string, numeric = false) => (
    <th
      scope="col"
      className={numeric ? 'num' : undefined}
      aria-sort={sort.key === key ? (sort.desc ? 'descending' : 'ascending') : 'none'}
    >
      <button
        type="button"
        onClick={() => setSort((s) => ({ key, desc: s.key === key ? !s.desc : numeric }))}
      >
        {label}
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
            {header('relationship', 'Relationship')}
            {header('value', measureLabel, true)}
            {header('papers', 'Papers', true)}
            {header('evidence', 'Evidence', true)}
            <th scope="col">Caveats</th>
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, MAX_ROWS).map(({ edge, agg, value }) => (
            <tr
              key={edge.id}
              className={edge.id === selectedPair ? 'is-selected' : undefined}
              tabIndex={0}
              onClick={() => onSelectPair(edge.source, edge.target)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault()
                  onSelectPair(edge.source, edge.target)
                }
              }}
            >
              <td>{name(edge.source)}</td>
              <td>{name(edge.target)}</td>
              <td className="nowrap">
                <span className={`line-key rel-${agg.relationship.replace(/ /g, '-')}`} aria-hidden />
                {capitalise(agg.relationship)}
              </td>
              <td className="num">{score(value)}</td>
              <td className="num">{agg.nPapers}</td>
              <td className="num">{agg.evidenceScore}/15</td>
              <td className="muted small">{agg.flags.join('; ')}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length > MAX_ROWS && (
        <p className="muted small table-note">
          Showing the first {MAX_ROWS} of {rows.length.toLocaleString()} pairs. Narrow the filters or export to CSV for
          the rest.
        </p>
      )}
    </div>
  )
}
