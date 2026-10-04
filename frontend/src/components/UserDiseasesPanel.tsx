import type { ApiHealth } from '../data/types'
import type { UserDisease } from '../lib/userDiseases'
import { InfoTip } from './InfoTip'

interface Props {
  diseases: UserDisease[]
  api: ApiHealth
  placing: Set<string>
  linksPerDisease: number
  selectedId: string | null
  onLinksChange: (n: number) => void
  onAdd: () => void
  onSelect: (id: string) => void
  onEdit: (disease: UserDisease) => void
  onRemove: (disease: UserDisease) => void
  onPlaceAll: () => void
  onExport: () => void
}

export function UserDiseasesPanel(props: Props) {
  const { diseases, api, placing } = props
  const unplaced = diseases.filter((d) => !d.placement).length
  return (
    <div className="user-panel">
      <div className="filters-header">
        <span className="label-row">
          <span className="section-label">Your diseases</span>
          <InfoTip topic="yourDiseases" />
        </span>
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
                    {d.input.name}
                    <span className="muted small">
                      {placing.has(d.id)
                        ? 'Placing…'
                        : d.placement
                          ? 'Placed'
                          : 'Not placed yet'}
                    </span>
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
            <span className="label-row">
              Links shown per added disease <InfoTip topic="linksPerDisease" />
            </span>
            <select value={props.linksPerDisease} onChange={(e) => props.onLinksChange(Number(e.target.value))}>
              {[3, 5, 10, 20].map((n) => (
                <option key={n} value={n}>
                  {n} most similar
                </option>
              ))}
            </select>
          </label>
          {api.status === 'ready' && (
            <button type="button" className="link-button small" onClick={props.onPlaceAll} disabled={placing.size > 0}>
              {unplaced > 0 ? `Place ${unplaced} unplaced` : 'Place all again'} (also matches them with each other)
            </button>
          )}
        </>
      )}

      {api.status !== 'ready' && (
        <p className="callout small">
          {api.status === 'loading'
            ? 'The placement service is loading the v2 model…'
            : api.status === 'error'
              ? `The placement service could not load the v2 model: ${api.error}`
              : 'To place new diseases, start the placement service: python frontend/api/server.py'}
        </p>
      )}
      <p className="muted small">Saved in this browser only. Export JSON to keep or share them.</p>
    </div>
  )
}
