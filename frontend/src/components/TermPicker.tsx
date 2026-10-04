import { useId, useMemo, useState } from 'react'
import { searchTerms, type TermOption } from '../data/annotations'

interface Props {
  label: string
  placeholder: string
  options: TermOption[]
  /** Chosen option ID, or null. */
  value: string | null
  onPick: (id: string) => void
  onClear: () => void
  /** Shown while the options are still loading. */
  loading?: boolean
  countLabel?: string
}

/** Type-ahead over a long vocabulary (12k HPO terms, 5k genes, 7.5k diseases). */
export function TermPicker({ label, placeholder, options, value, onPick, onClear, loading, countLabel = 'diseases' }: Props) {
  const id = useId()
  const [text, setText] = useState('')
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(0)
  const matches = useMemo(() => (open ? searchTerms(options, text, 8) : []), [options, text, open])
  const chosen = value ? options.find((o) => o.id === value) : undefined

  const pick = (option: TermOption) => {
    onPick(option.id)
    setText('')
    setOpen(false)
    setActive(0)
  }

  return (
    <div className="term-picker">
      <label htmlFor={id} className="form-label">
        {label}
      </label>
      {value ? (
        <div className="picked-term">
          <span>
            <strong>{chosen?.label ?? value}</strong>
            {chosen && chosen.label !== chosen.id && <span className="muted small"> {chosen.id}</span>}
            {chosen && chosen.count > 0 && (
              <span className="muted small">
                {' '}
                · {chosen.count.toLocaleString()} {countLabel}
              </span>
            )}
          </span>
          <button type="button" className="link-button small" onClick={onClear}>
            Clear
          </button>
        </div>
      ) : (
        <div className="picker-box">
          <input
            id={id}
            type="text"
            role="combobox"
            aria-expanded={open && matches.length > 0}
            aria-controls={`${id}-list`}
            autoComplete="off"
            value={text}
            disabled={loading}
            placeholder={loading ? 'Loading annotations…' : placeholder}
            onChange={(e) => {
              setText(e.target.value)
              setOpen(true)
              setActive(0)
            }}
            onFocus={() => setOpen(true)}
            onBlur={() => setTimeout(() => setOpen(false), 120)}
            onKeyDown={(e) => {
              if (e.key === 'ArrowDown') {
                e.preventDefault()
                setActive((a) => Math.min(a + 1, matches.length - 1))
              } else if (e.key === 'ArrowUp') {
                e.preventDefault()
                setActive((a) => Math.max(a - 1, 0))
              } else if (e.key === 'Enter' && matches[active]) {
                e.preventDefault()
                pick(matches[active])
              } else if (e.key === 'Escape') {
                setOpen(false)
              }
            }}
          />
          {open && matches.length > 0 && (
            <ul id={`${id}-list`} role="listbox" className="picker-list">
              {matches.map((m, i) => (
                <li key={m.id} role="option" aria-selected={i === active}>
                  <button type="button" tabIndex={-1} onMouseDown={(e) => e.preventDefault()} onClick={() => pick(m)}>
                    <span>{m.label}</span>
                    <span className="muted small">
                      {m.label !== m.id ? `${m.id} · ` : ''}
                      {m.count.toLocaleString()}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}
