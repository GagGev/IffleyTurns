import type { DatasetKey, LiteratureGraph, Relationship } from '../data/types'
import { DIMENSIONS, DIMENSION_LABELS, RELATIONSHIPS } from '../data/types'
import { DEFAULT_FILTERS, type Filters, type Measure, type Rarity } from '../lib/graph'
import { capitalise } from '../lib/format'

interface Props {
  graph: LiteratureGraph
  filters: Filters
  onChange: (next: Filters) => void
}

function toggle<T>(list: T[], value: T): T[] {
  return list.includes(value) ? list.filter((v) => v !== value) : [...list, value]
}

export function FilterPanel({ graph, filters, onChange }: Props) {
  const set = <K extends keyof Filters>(key: K, value: Filters[K]) => onChange({ ...filters, [key]: value })
  const isDefault = JSON.stringify(filters) === JSON.stringify(DEFAULT_FILTERS)

  return (
    <div className="filters">
      <div className="filters-header">
        <span className="section-label">Filters</span>
        <button type="button" className="link-button" disabled={isDefault} onClick={() => onChange(DEFAULT_FILTERS)}>
          Reset
        </button>
      </div>

      <label className="field">
        <span>Score by</span>
        <select value={filters.measure} onChange={(e) => set('measure', e.target.value as Measure)}>
          <option value="overall">Overall similarity</option>
          <optgroup label="Single dimension">
            {DIMENSIONS.map((d) => (
              <option key={d} value={d}>
                {DIMENSION_LABELS[d]}
              </option>
            ))}
          </optgroup>
        </select>
        <small>
          {filters.measure === 'overall'
            ? 'Evidence-weighted mean of paper scores.'
            : 'Only pairs with papers scoring this dimension are shown.'}
        </small>
      </label>

      <label className="field">
        <span>
          Minimum score <output>{filters.minScore.toFixed(2)}</output>
        </span>
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={filters.minScore}
          onChange={(e) => set('minScore', Number(e.target.value))}
        />
      </label>

      <label className="field">
        <span>
          Minimum evidence strength <output>{Math.round(filters.minEvidence * 15)}/15</output>
        </span>
        <input
          type="range"
          min={0}
          max={1}
          step={1 / 15}
          value={filters.minEvidence}
          onChange={(e) => set('minEvidence', Number(e.target.value))}
        />
        <small>Study design, sample size and replication.</small>
      </label>

      <label className="field">
        <span>Minimum papers per pair</span>
        <select value={filters.minPapers} onChange={(e) => set('minPapers', Number(e.target.value))}>
          {[1, 2, 3, 5].map((n) => (
            <option key={n} value={n}>
              {n === 1 ? 'Any' : `${n} or more`}
            </option>
          ))}
        </select>
      </label>

      <fieldset className="field">
        <legend>Relationship</legend>
        {RELATIONSHIPS.map((r: Relationship) => (
          <label key={r} className="check">
            <input
              type="checkbox"
              checked={filters.relationships.includes(r)}
              onChange={() => set('relationships', toggle(filters.relationships, r))}
            />
            <span className={`line-key rel-${r.replace(/ /g, '-')}`} aria-hidden />
            {capitalise(r)}
          </label>
        ))}
      </fieldset>

      <label className="field">
        <span>Rarity</span>
        <select value={filters.rarity} onChange={(e) => set('rarity', e.target.value as Rarity)}>
          <option value="any">Any pair</option>
          <option value="one-rare">At least one rare disease</option>
          <option value="both-rare">Both diseases rare</option>
        </select>
      </label>

      <fieldset className="field">
        <legend>Evidence source</legend>
        {graph.datasets.map((d) => (
          <label key={d.key} className="check">
            <input
              type="checkbox"
              checked={filters.datasets.includes(d.key)}
              onChange={() => set('datasets', toggle(filters.datasets, d.key as DatasetKey))}
            />
            {d.label}
          </label>
        ))}
        {filters.datasets.length === 0 && <small className="warning-text">Select at least one source.</small>}
      </fieldset>
    </div>
  )
}
