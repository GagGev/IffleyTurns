import { useId, useMemo, useState } from 'react'
import type { Disease } from '../data/types'
import { displayName, plural } from '../lib/format'
import { searchKey } from '../lib/graph'

interface Props {
  diseases: Disease[]
  onSelect: (id: string) => void
}

interface Indexed {
  disease: Disease
  name: string
  aliases: string[]
  orpha: string
}

const MAX_RESULTS = 10

export function SearchBox({ diseases, onSelect }: Props) {
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(0)
  const listId = useId()

  const index = useMemo<Indexed[]>(
    () =>
      diseases.map((d) => ({
        disease: d,
        name: searchKey(d.name),
        aliases: d.aliases.map(searchKey),
        orpha: [d.orphaId, ...d.candidateOrphaIds].filter(Boolean).join(' ').toLowerCase(),
      })),
    [diseases],
  )

  const results = useMemo(() => {
    const q = searchKey(query.trim())
    if (!q) return []
    const numeric = q.replace(/^orpha:?/, '')
    const scored: { d: Disease; rank: number; via?: string }[] = []
    for (const item of index) {
      let rank = -1
      let via: string | undefined
      if (item.name.startsWith(q)) rank = 0
      else if (item.name.includes(q)) rank = 1
      else {
        const alias = item.aliases.findIndex((a) => a.includes(q))
        if (alias >= 0) {
          rank = 2
          via = item.disease.aliases[alias]
        } else if (/^\d+$/.test(numeric) && item.orpha.split(' ').some((id) => id === `orpha:${numeric}`)) {
          rank = 0
        }
      }
      if (rank >= 0) scored.push({ d: item.disease, rank, via })
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
      <label htmlFor={`${listId}-input`} className="section-label">
        Find a disease
      </label>
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
          {results.map(({ d, via }, i) => (
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
                {d.origin === 'user'
                  ? 'Added by you'
                  : d.origin === 'catalogue'
                    ? `${d.orphaId} · feature profile only`
                    : `${d.orphaId ?? 'no ORPHA ID'} · ${plural(d.degree, 'pair')}`}
                {via && <> · also “{via}”</>}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
