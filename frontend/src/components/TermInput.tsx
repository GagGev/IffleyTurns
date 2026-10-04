import { useEffect, useId, useState } from 'react'
import { suggest, type SuggestField } from '../data/source'
import { FREQUENCIES } from '../lib/userDiseases'

interface Props {
  label: string
  hint: string
  placeholder: string
  values: string[]
  onChange: (values: string[]) => void
  /** Vocabulary to suggest from (needs the placement service). */
  field?: SuggestField
  /** Turn a typed token into the stored value, or return an error message. */
  normalise?: (token: string) => string | { error: string }
  /** Readable names for stored IDs. */
  labels: Record<string, string>
  onLabel: (id: string, label: string) => void
  /** Per-value weight (phenotype frequency). */
  weights?: Record<string, number>
  onWeight?: (value: string, weight: number) => void
}

/** Chip input: free entry, pasted lists, and suggestions from v2's vocabulary. */
export function TermInput(props: Props) {
  const { values, onChange, field, labels } = props
  const id = useId()
  const [text, setText] = useState('')
  const [problem, setProblem] = useState<string | null>(null)
  const [active, setActive] = useState(0)
  const [suggestions, setSuggestions] = useState<{ q: string; items: { id: string; label: string }[] }>({ q: '', items: [] })

  const query = text.trim()
  useEffect(() => {
    if (!field || query.length < 2) return
    let cancelled = false
    const timer = setTimeout(() => {
      suggest(field, query).then((items) => !cancelled && setSuggestions({ q: query, items }))
    }, 150)
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [field, query])
  const shown = suggestions.q === query && query.length >= 2 ? suggestions.items.filter((s) => !values.includes(s.id)) : []

  const commit = (raw: string, label?: string) => {
    const tokens = label ? [raw] : raw.split(/[,;\n\t]+/).map((t) => t.trim()).filter(Boolean)
    if (tokens.length === 0) return
    const added: string[] = []
    const rejected: string[] = []
    for (const token of tokens) {
      const result = props.normalise ? props.normalise(token) : token
      if (typeof result === 'string') added.push(result)
      else rejected.push(`${token}: ${result.error}`)
    }
    if (label && added[0]) props.onLabel(added[0], label)
    onChange([...new Set([...values, ...added])])
    setProblem(rejected.length ? rejected.join('; ') : null)
    setText('')
    setActive(0)
  }

  return (
    <div className="term-input">
      <label htmlFor={id} className="form-label">
        {props.label} <span className="optional">optional</span>
      </label>
      <div className="chip-box">
        {values.map((v) => (
          <span key={v} className="term-chip" title={labels[v] ?? v}>
            <span>
              {labels[v] ? (
                <>
                  {labels[v]} <span className="term-name">{v}</span>
                </>
              ) : (
                v
              )}
            </span>
            {props.weights && props.onWeight && (
              <select
                aria-label={`How often ${labels[v] ?? v} occurs`}
                value={FREQUENCIES.find((f) => Math.abs(f.weight - (props.weights![v] ?? 0.5)) < 0.01)?.weight ?? 0.5}
                onChange={(e) => props.onWeight!(v, Number(e.target.value))}
              >
                {FREQUENCIES.map((f) => (
                  <option key={f.label} value={f.weight}>
                    {f.label}
                  </option>
                ))}
              </select>
            )}
            <button type="button" aria-label={`Remove ${labels[v] ?? v}`} onClick={() => onChange(values.filter((x) => x !== v))}>
              ×
            </button>
          </span>
        ))}
        <input
          id={id}
          value={text}
          placeholder={values.length ? '' : props.placeholder}
          role="combobox"
          aria-expanded={shown.length > 0}
          aria-controls={`${id}-list`}
          aria-describedby={`${id}-hint`}
          autoComplete="off"
          onChange={(e) => {
            setText(e.target.value)
            setActive(0)
          }}
          onPaste={(e) => {
            const pasted = e.clipboardData.getData('text')
            if (/[,;\n\t]/.test(pasted)) {
              e.preventDefault()
              commit(text + pasted)
            }
          }}
          onKeyDown={(e) => {
            if (e.key === 'ArrowDown' && shown.length) {
              e.preventDefault()
              setActive((a) => Math.min(a + 1, shown.length - 1))
            } else if (e.key === 'ArrowUp' && shown.length) {
              e.preventDefault()
              setActive((a) => Math.max(a - 1, 0))
            } else if (e.key === 'Enter' || e.key === ',') {
              if (!text.trim()) return
              e.preventDefault()
              const pick = e.key === 'Enter' ? shown[active] : undefined
              if (pick) commit(pick.id, pick.label || undefined)
              else commit(text)
            } else if (e.key === 'Backspace' && !text && values.length) {
              onChange(values.slice(0, -1))
            }
          }}
          onBlur={() => text.trim() && commit(text)}
        />
      </div>
      {shown.length > 0 && (
        <ul className="term-suggestions" role="listbox" id={`${id}-list`}>
          {shown.map((s, i) => (
            <li
              key={s.id}
              role="option"
              aria-selected={i === active}
              className={i === active ? 'active' : undefined}
              onMouseDown={(e) => {
                e.preventDefault()
                commit(s.id, s.label || undefined)
              }}
            >
              <span className="term-id">{s.id}</span> {s.label}
            </li>
          ))}
        </ul>
      )}
      <small id={`${id}-hint`} className={problem ? 'warning-text' : 'muted'}>
        {problem ?? props.hint}
      </small>
    </div>
  )
}
