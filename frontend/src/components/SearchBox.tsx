import { useId, useMemo, useState } from 'react'
import type { GraphNode } from '../data/types'
import { clusterId, displayName, plural } from '../lib/format'
import { searchKey } from '../lib/graph'
import { InfoTip } from './InfoTip'

interface Props {
  diseases: GraphNode[]
  onSelect: (id: string) => void
}

interface Indexed {
  disease: GraphNode
  name: string
  id: string
}

const MAX_RESULTS = 10

export function SearchBox({ diseases, onSelect }: Props) {
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(0)
  const listId = useId()

  const index = useMemo<Indexed[]>(
    () =>
      diseases.map((d) => ({ disease: d, name: searchKey(d.name), id: d.id.toLowerCase() })),
    [diseases],
  )

  const results = useMemo(() => {
    const q = searchKey(query.trim())
    if (!q) return []
    const numeric = q.replace(/^orpha:?/, '')
    const scored: { d: GraphNode; rank: number }[] = []
    const wordStart = new RegExp(`\\b${q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`)
    for (const item of index) {
      let rank = -1
      if (/^\d+$/.test(numeric) && item.id === `orpha:${numeric}`) rank = 0
      else if (item.name.startsWith(q)) rank = 0
      else if (wordStart.test(item.name)) rank = 1
      else if (item.name.includes(q)) rank = 2
      if (rank >= 0) scored.push({ d: item.disease, rank })
    }
    scored.sort((x, y) => x.rank - y.rank || y.d.degree - x.d.degree || x.d.name.localeCompare(y.d.name))
    return scored.slice(0, MAX_RESULTS)
  }, [query, index])

  const choose = (id: string) => {
    onSelect(id)
    setQuery('')
    setOpen(false)
  }

  return (
    <div className="search">
      <div className="label-row">
        <label htmlFor={`${listId}-input`} className="section-label">
          Find a disease
        </label>
        <InfoTip topic="search" />
      </div>
      <input
        id={`${listId}-input`}
        type="search"
        role="combobox"
        aria-expanded={open && results.length > 0}
        aria-controls={listId}
        aria-activedescendant={results[active] ? `${listId}-${active}` : undefined}
        placeholder="Name, synonym or ORPHA ID"
        value={query}
        autoComplete="off"
        onChange={(e) => {
          setQuery(e.target.value)
          setActive(0)
          setOpen(true)
        }}
        onFocus={() => setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 120)}
        onKeyDown={(e) => {
          if (e.key === 'ArrowDown') {
            e.preventDefault()
            setActive((a) => Math.min(a + 1, results.length - 1))
          } else if (e.key === 'ArrowUp') {
            e.preventDefault()
            setActive((a) => Math.max(a - 1, 0))
          } else if (e.key === 'Enter' && results[active]) {
            choose(results[active].d.id)
          } else if (e.key === 'Escape') {
            setOpen(false)
          }
        }}
      />
      {open && query.trim() && (
        <ul className="search-results" role="listbox" id={listId}>
          {results.length === 0 && <li className="search-empty">No matching diseases.</li>}
          {results.map(({ d }, i) => (
            <li
              key={d.id}
              id={`${listId}-${i}`}
              role="option"
              aria-selected={i === active}
              className={i === active ? 'active' : undefined}
              onMouseDown={(e) => {
                e.preventDefault()
                choose(d.id)
              }}
              onMouseEnter={() => setActive(i)}
            >
              <span className="search-name">{displayName(d.name)}</span>
              <span className="search-meta">
                {d.origin === 'user' ? 'Added by you' : `${d.id}${d.cluster >= 0 ? ` · ${clusterId(d.cluster)}` : ''} · ${d.category || d.disorderType}`} ·{' '}
                {plural(d.degree, 'edge')}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
