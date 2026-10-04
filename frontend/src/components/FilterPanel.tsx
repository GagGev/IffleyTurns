import type { GraphData, Support } from '../data/types'
import { SUPPORT_LEVELS, SUPPORT_MEANING } from '../data/types'
import { DEFAULT_FILTERS, type Filters, PERCENTILE_STEPS } from '../lib/graph'
import { capitalise, modalityLabel, percentile } from '../lib/format'
import { InfoTip } from './InfoTip'

interface Props {
  graph: GraphData
  filters: Filters
  onChange: (next: Filters) => void
}

export function FilterPanel({ graph, filters, onChange }: Props) {
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
    </div>
  )
}
