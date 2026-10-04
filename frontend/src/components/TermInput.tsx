import { useId, useMemo, useState } from 'react'
import type { FeatureCatalogue, FeatureFamily } from '../data/features'
import { resolveTerm } from '../lib/userDiseases'

interface Props {
  family: FeatureFamily
  label: string
  hint: string
  placeholder: string
  values: string[]
  onChange: (values: string[]) => void
  catalogue: FeatureCatalogue | null
}

interface Suggestion {
  id: string
  name: string
  key: string
}

const suggestionCache = new WeakMap<FeatureCatalogue, Map<FeatureFamily, Suggestion[]>>()

function suggestionsFor(catalogue: FeatureCatalogue, family: FeatureFamily): Suggestion[] {
  let perFamily = suggestionCache.get(catalogue)
  if (!perFamily) {
    perFamily = new Map()
    suggestionCache.set(catalogue, perFamily)
  }
  let list = perFamily.get(family)
  if (!list) {
    const counts = new Map<string, number>()
    for (const p of catalogue.profiles) for (const t of p.sets[family] ?? []) counts.set(t, (counts.get(t) ?? 0) + 1)
    const names = catalogue.termNames[family]
    // Most widely used terms first, so common choices surface early.
    list = [...counts.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([id]) => ({ id, name: names.get(id) ?? '', key: `${id} ${names.get(id) ?? ''}`.toLowerCase() }))
    perFamily.set(family, list)
  }
  return list
}

/** Chip input for one feature family: free entry, paste of lists, and catalogue suggestions. */
export function TermInput({ family, label, hint, placeholder, values, onChange, catalogue }: Props) {
  const id = useId()
  const [text, setText] = useState('')
  const [problem, setProblem] = useState<string | null>(null)
  const [active, setActive] = useState(0)

  const suggestions = useMemo(() => {
    const q = text.trim().toLowerCase()
    if (!catalogue || q.length < 2) return []
    const out: Suggestion[] = []
    for (const s of suggestionsFor(catalogue, family)) {
      if (s.key.includes(q) && !values.includes(s.id)) out.push(s)
      if (out.length === 8) break
    }
    return out
  }, [text, catalogue, family, values])

  const commit = (raw: string) => {
    const tokens = raw.split(/[,;\n\t]+/).map((t) => t.trim()).filter(Boolean)
    if (tokens.length === 0) return
    const added: string[] = []
    const rejected: string[] = []
    for (const token of tokens) {
      const check = resolveTerm(family, token, catalogue)
      if (check.value) added.push(check.value)
      else rejected.push(`${token}: ${check.message}`)
    }
    onChange([...new Set([...values, ...added])])
    setProblem(rejected.length ? rejected.join('; ') : null)
    setText('')
    setActive(0)
  }

  const nameOf = (term: string) => catalogue?.termNames[family].get(term)
  const known = (term: string) => !catalogue || resolveTerm(family, term, catalogue).known

  return (
    <div className="term-input">
      <label htmlFor={id} className="form-label">
        {label} <span className="optional">optional</span>
      </label>
      <div className="chip-box">
        {values.map((v) => (
          <span key={v} className={`term-chip${known(v) ? '' : ' is-unknown'}`} title={known(v) ? nameOf(v) : 'Not found in any catalogue disease'}>
            <span>
              {v}
              {nameOf(v) && <span className="term-name"> {nameOf(v)}</span>}
            </span>
            <button type="button" aria-label={`Remove ${v}`} onClick={() => onChange(values.filter((x) => x !== v))}>
              ×
            </button>
          </span>
        ))}
        <input
          id={id}
          value={text}
          placeholder={values.length ? '' : placeholder}
          role="combobox"
          aria-expanded={suggestions.length > 0}
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
            if (e.key === 'ArrowDown' && suggestions.length) {
              e.preventDefault()
              setActive((a) => Math.min(a + 1, suggestions.length - 1))
            } else if (e.key === 'ArrowUp' && suggestions.length) {
              e.preventDefault()
              setActive((a) => Math.max(a - 1, 0))
            } else if (e.key === 'Enter' || e.key === ',') {
              if (!text.trim()) return
              e.preventDefault()
              if (e.key === 'Enter' && suggestions[active]) commit(suggestions[active].id)
              else commit(text)
            } else if (e.key === 'Backspace' && !text && values.length) {
              onChange(values.slice(0, -1))
            }
          }}
          onBlur={() => text.trim() && commit(text)}
        />
      </div>
      {suggestions.length > 0 && (
        <ul className="term-suggestions" role="listbox" id={`${id}-list`}>
          {suggestions.map((s, i) => (
            <li
              key={s.id}
              role="option"
              aria-selected={i === active}
              className={i === active ? 'active' : undefined}
              onMouseDown={(e) => {
                e.preventDefault()
                commit(s.id)
              }}
            >
              <span className="term-id">{s.id}</span> {s.name}
            </li>
          ))}
        </ul>
      )}
      <small id={`${id}-hint`} className={problem ? 'warning-text' : 'muted'}>
        {problem ?? hint}
      </small>
    </div>
  )
}
