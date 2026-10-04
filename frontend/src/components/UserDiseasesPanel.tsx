import type { FeatureCatalogue } from '../data/features'
import { plural } from '../lib/format'
import { featureCount } from '../lib/similarity'
import { toProfile, type UserDisease } from '../lib/userDiseases'

interface Props {
  diseases: UserDisease[]
  catalogue: FeatureCatalogue | null | undefined
  linksPerDisease: number
  selectedId: string | null
  onLinksChange: (n: number) => void
  onAdd: () => void
  onSelect: (id: string) => void
  onEdit: (disease: UserDisease) => void
  onRemove: (disease: UserDisease) => void
  onExport: () => void
}

export function UserDiseasesPanel(props: Props) {
  const { diseases, catalogue } = props
  return (
    <div className="user-panel">
      <div className="filters-header">
        <span className="section-label">Your diseases</span>
        {diseases.length > 0 && (
          <button type="button" className="link-button small" onClick={props.onExport}>
            Export JSON
          </button>
        )}
      </div>
      <button type="button" className="button primary full" onClick={props.onAdd}>
        + Add a disease
      </button>

      {diseases.length > 0 && (
        <>
          <ul className="user-list">
            {diseases.map((d) => (
              <li key={d.id} className={d.id === props.selectedId ? 'is-selected' : undefined}>
                <button type="button" className="user-name" onClick={() => props.onSelect(d.id)}>
                  <span className="user-mark" aria-hidden>
                    ◆
                  </span>
                  <span className="user-name-text">
                    {d.name}
                    <span className="muted small">{plural(featureCount(toProfile(d, catalogue ?? null)), 'feature')}</span>
                  </span>
                </button>
                <span className="user-actions">
                  <button type="button" className="link-button small" onClick={() => props.onEdit(d)}>
                    Edit
                  </button>
                  <button type="button" className="link-button small" onClick={() => props.onRemove(d)}>
                    Remove
                  </button>
                </span>
              </li>
            ))}
          </ul>
          <label className="field">
            <span>Similarity links per added disease</span>
            <select value={props.linksPerDisease} onChange={(e) => props.onLinksChange(Number(e.target.value))}>
              {[3, 5, 10, 20].map((n) => (
                <option key={n} value={n}>
                  {n} most similar
                </option>
              ))}
            </select>
          </label>
        </>
      )}

      {catalogue === null && (
        <p className="callout small">
          Feature profiles for catalogue diseases have not been generated, so added diseases can only be compared with
          each other for now. Run <code>generate_features.py</code>, then{' '}
          <code>npm run features</code>, to compare against every Orphanet disease.
        </p>
      )}
      <p className="muted small">Saved in this browser only. Use Export JSON to keep or share them.</p>
    </div>
  )
}
