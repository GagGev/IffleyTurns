import { useMemo } from 'react'
import type { TermOption } from '../data/annotations'
import type { GraphData, Support } from '../data/types'
import { SUPPORT_LEVELS, SUPPORT_MEANING } from '../data/types'
import { DEFAULT_FILTERS, type Filters, PERCENTILE_STEPS } from '../lib/graph'
import { capitalise, modalityLabel, percentile } from '../lib/format'
import { InfoTip } from './InfoTip'
import { TermPicker } from './TermPicker'

interface Props {
  graph: GraphData
  filters: Filters
  onChange: (next: Filters) => void
  /** HPO terms to pick from; null until the annotations have loaded. */
  symptomOptions: TermOption[] | null
  symptomError: string | null
  /** How many diseases match the chosen symptoms, or null when none are chosen. */
  symptomMatches: number | null
}

export function FilterPanel({ graph, filters, onChange, symptomOptions, symptomError, symptomMatches }: Props) {
  const set = <K extends keyof Filters>(key: K, value: Filters[K]) => onChange({ ...filters, [key]: value })
  const isDefault = JSON.stringify(filters) === JSON.stringify(DEFAULT_FILTERS)
  const toggleSupport = (s: Support) =>
    set('support', filters.support.includes(s) ? filters.support.filter((x) => x !== s) : [...filters.support, s])

  return (
    <div className="filters">
      <div className="filters-header">
        <span className="label-row">
          <span className="section-label">Filters</span>
          <InfoTip topic="filters" />
        </span>
        <button type="button" className="link-button" disabled={isDefault} onClick={() => onChange(DEFAULT_FILTERS)}>
          Reset
        </button>
      </div>

      <fieldset className="field">
        <legend className="label-row">
          Edge support <InfoTip topic="support" />
        </legend>
        {SUPPORT_LEVELS.map((s) => (
          <label key={s} className="check" title={SUPPORT_MEANING[s]}>
            <input type="checkbox" checked={filters.support.includes(s)} onChange={() => toggleSupport(s)} />
            <span className={`line-key support-${s}`} aria-hidden />
            {capitalise(s)}
            <span className="muted small">{(graph.stats.support[s] ?? 0).toLocaleString()}</span>
          </label>
        ))}
      </fieldset>

      <div className="label-row">
        <label className="check">
          <input type="checkbox" checked={filters.mutualOnly} onChange={(e) => set('mutualOnly', e.target.checked)} />
          Mutual neighbours only
        </label>
        <InfoTip topic="mutual" />
      </div>

      <label className="field">
        <span className="label-row">
          Minimum percentile <InfoTip topic="percentile" />
        </span>
        <select value={filters.minPercentile} onChange={(e) => set('minPercentile', Number(e.target.value))}>
          {PERCENTILE_STEPS.map((p) => (
            <option key={p} value={p}>
              {p === 0 ? 'Any' : `≥ ${percentile(p)} of random pairs`}
            </option>
          ))}
        </select>
      </label>

      <label className="field">
        <span className="label-row">
          Main evidence <InfoTip topic="mainEvidence" />
        </span>
        <select value={filters.mainModality ?? ''} onChange={(e) => set('mainModality', e.target.value || null)}>
          <option value="">Any modality</option>
          {graph.modalities.map((m) => (
            <option key={m} value={m}>
              {modalityLabel(m)}
            </option>
          ))}
        </select>
      </label>

      {graph.clusters.length > 0 && (
        <label className="field">
          <span className="label-row">
            Cluster <InfoTip topic="cluster" />
          </span>
          <select
            value={filters.cluster ?? ''}
            onChange={(e) => set('cluster', e.target.value === '' ? null : Number(e.target.value))}
          >
            <option value="">All clusters</option>
            {graph.clusters.map((c, i) => (
              <option key={c.id} value={i}>
                {c.id} · {c.label} ({c.size.toLocaleString()})
              </option>
            ))}
          </select>
        </label>
      )}

      <label className="field">
        <span className="label-row">
          Orphanet category <InfoTip topic="category" />
        </span>
        <select value={filters.category ?? ''} onChange={(e) => set('category', e.target.value || null)}>
          <option value="">All categories</option>
          {graph.categories.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
      </label>
      <div className="label-row">
        <label className="check">
          <input
            type="checkbox"
            checked={filters.symptomFilter}
            onChange={(e) => set('symptomFilter', e.target.checked)}
          />
          Filter by symptoms
        </label>
        <InfoTip topic="symptoms" />
      </div>
      {filters.symptomFilter && (
        <SymptomFilter
          filters={filters}
          onChange={onChange}
          options={symptomOptions}
          error={symptomError}
          matches={symptomMatches}
        />
      )}
    </div>
  )
}

function SymptomFilter({
  filters,
  onChange,
  options,
  error,
  matches,
}: {
  filters: Filters
  onChange: (next: Filters) => void
  options: TermOption[] | null
  error: string | null
  matches: number | null
}) {
  const labels = useMemo(() => new Map((options ?? []).map((o) => [o.id, o.label])), [options])
  const set = (symptoms: string[]) => onChange({ ...filters, symptoms })
  const chosen = filters.symptoms
  return (
    <div className="symptom-filter">
      {chosen.length > 0 && (
        <ul className="symptom-chips" aria-label="Chosen symptoms">
          {chosen.map((id) => (
            <li key={id}>
              <span title={id}>{labels.get(id) ?? id}</span>
              <button
                type="button"
                aria-label={`Remove ${labels.get(id) ?? id}`}
                onClick={() => set(chosen.filter((x) => x !== id))}
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      )}
      {error ? (
        <p className="callout small">{error}</p>
      ) : (
        <TermPicker
          label={chosen.length ? 'Add another symptom' : 'Add a symptom'}
          placeholder="e.g. seizure, HP:0001250"
          options={(options ?? []).filter((o) => !chosen.includes(o.id))}
          value={null}
          loading={options === null}
          onPick={(id) => set([...chosen, id])}
          onClear={() => {}}
        />
      )}
      {chosen.length >= 2 && (
        <div className="radio-row small" role="radiogroup" aria-label="How to combine symptoms">
          <label className="check">
            <input
              type="radio"
              checked={filters.symptomMatch === 'all'}
              onChange={() => onChange({ ...filters, symptomMatch: 'all' })}
            />
            Has all of them
          </label>
          <label className="check">
            <input
              type="radio"
              checked={filters.symptomMatch === 'any'}
              onChange={() => onChange({ ...filters, symptomMatch: 'any' })}
            />
            Has any of them
          </label>
        </div>
      )}
      {matches !== null && (
        <small className="muted">
          {matches.toLocaleString()} {matches === 1 ? 'disease has' : 'diseases have'}{' '}
          {chosen.length === 1 ? 'this symptom' : filters.symptomMatch === 'all' ? 'all of these' : 'at least one of these'}.
        </small>
      )}
      {chosen.length === 0 && !error && <small className="muted">Choose a symptom to limit the graph.</small>}
    </div>
  )
}
