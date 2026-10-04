import type { GraphData, Support } from '../data/types'
import { SUPPORT_LEVELS, SUPPORT_MEANING } from '../data/types'
import { DEFAULT_FILTERS, type Filters, PERCENTILE_STEPS } from '../lib/graph'
import { capitalise, modalityLabel, percentile } from '../lib/format'

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
        <span className="section-label">Filters</span>
        <button type="button" className="link-button" disabled={isDefault} onClick={() => onChange(DEFAULT_FILTERS)}>
          Reset
        </button>
      </div>

      <fieldset className="field">
        <legend>Edge support</legend>
        {SUPPORT_LEVELS.map((s) => (
          <label key={s} className="check" title={SUPPORT_MEANING[s]}>
            <input type="checkbox" checked={filters.support.includes(s)} onChange={() => toggleSupport(s)} />
            <span className={`line-key support-${s}`} aria-hidden />
            {capitalise(s)}
            <span className="muted small">{(graph.stats.support[s] ?? 0).toLocaleString()}</span>
          </label>
        ))}
        <small>Novel edges are hypotheses with no curated or shared-annotation backing.</small>
      </fieldset>

      <label className="check">
        <input type="checkbox" checked={filters.mutualOnly} onChange={(e) => set('mutualOnly', e.target.checked)} />
        Mutual neighbours only
      </label>
      <small className="muted filter-note">
        Both diseases list each other among their top {graph.k ?? 10}. Turn off to see all{' '}
        {graph.stats.edges.toLocaleString()} edges.
      </small>

      <label className="field">
        <span>Minimum percentile</span>
        <select value={filters.minPercentile} onChange={(e) => set('minPercentile', Number(e.target.value))}>
          {PERCENTILE_STEPS.map((p) => (
            <option key={p} value={p}>
              {p === 0 ? 'Any' : `≥ ${percentile(p)} of random pairs`}
            </option>
          ))}
        </select>
      </label>

      <label className="field">
        <span>Main evidence</span>
        <select value={filters.mainModality ?? ''} onChange={(e) => set('mainModality', e.target.value || null)}>
          <option value="">Any modality</option>
          {graph.modalities.map((m) => (
            <option key={m} value={m}>
              {modalityLabel(m)}
            </option>
          ))}
        </select>
        <small>The modality contributing most to the score. Try Drugs or Drug targets for repurposing leads.</small>
      </label>

      {graph.clusters.length > 0 && (
        <label className="field">
          <span>Cluster</span>
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
          <small>Shows edges with both diseases in this cluster.</small>
        </label>
      )}

      <label className="field">
        <span>Orphanet category</span>
        <select value={filters.category ?? ''} onChange={(e) => set('category', e.target.value || null)}>
          <option value="">All categories</option>
          {graph.categories.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
        <small>Shows edges with both diseases in this category.</small>
      </label>
    </div>
  )
}
