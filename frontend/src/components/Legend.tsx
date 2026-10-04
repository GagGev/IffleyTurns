import { useMemo } from 'react'
import type { TermOption } from '../data/annotations'
import type { ClusterInfo, GraphNode } from '../data/types'
import { COLOUR_MODES, type NodeColouring } from '../lib/colouring'
import { plural } from '../lib/format'
import { TermPicker } from './TermPicker'

interface Props {
  colouring: NodeColouring
  clusters: ClusterInfo[]
  modularity: number | null
  categories: { name: string; count: number }[]
  /** Every disease in view, for the "similarity to a disease" picker. */
  nodes: Map<string, GraphNode>
  clusterColour: (index: number) => string
  categoryColour: (name: string) => string
  /** Cluster currently highlighted in the graph. */
  focusCluster: number | null
  onFocusCluster: (next: number | null) => void
}

export function Legend({
  colouring,
  clusters,
  modularity,
  categories,
  nodes,
  clusterColour,
  categoryColour,
  focusCluster,
  onFocusCluster,
}: Props) {
  const { mode, annotations } = colouring
  const modes = clusters.length > 0 ? COLOUR_MODES : COLOUR_MODES.filter((m) => m.value !== 'cluster')
  const hint = COLOUR_MODES.find((m) => m.value === mode)?.hint
  const diseaseOptions = useMemo<TermOption[]>(
    () => (mode === 'disease' ? [...nodes.values()].map((n) => ({ id: n.id, label: n.name, count: n.degree })) : []),
    [mode, nodes],
  )

  return (
    <div className="legend">
      <label className="field">
        <span className="section-label">Node colour</span>
        <select value={mode} onChange={(e) => colouring.setMode(e.target.value as typeof mode)}>
          {modes.map((m) => (
            <option key={m.value} value={m.value}>
              {m.label}
            </option>
          ))}
        </select>
      </label>
      {hint && <p className="muted small">{hint}.</p>}

      {mode === 'cluster' && clusters.length > 0 && (
        <>
          <p className="muted small">
            {plural(clusters.length, 'cluster')} found with Louvain community detection on the similarity graph
            {modularity !== null ? ` (modularity ${modularity.toFixed(2)})` : ''}. Click one to highlight it.
          </p>
          <ul className="cluster-list">
            {clusters.map((c, i) => (
              <li key={c.id}>
                <button
                  type="button"
                  className="cluster-item"
                  aria-pressed={focusCluster === i}
                  title={`${c.id}: ${c.size.toLocaleString()} diseases, ${Math.round(c.purity * 100)}% ${c.topCategory}. Hub: ${c.hubName}`}
                  onClick={() => onFocusCluster(focusCluster === i ? null : i)}
                >
                  <span className="swatch" style={{ background: clusterColour(i) }} aria-hidden />
                  <span className="cluster-name">
                    <strong>{c.id}</strong> {c.label}
                  </span>
                  <span className="muted small">{c.size.toLocaleString()}</span>
                </button>
              </li>
            ))}
          </ul>
        </>
      )}

      {mode === 'category' && (
        <ul className="cluster-list">
          {categories.map((c) => (
            <li key={c.name} className="cluster-item is-static">
              <span className="swatch" style={{ background: categoryColour(c.name) }} aria-hidden />
              <span className="cluster-name">{c.name}</span>
              <span className="muted small">{c.count.toLocaleString()}</span>
            </li>
          ))}
        </ul>
      )}

      {mode === 'gene' && (
        <TermPicker
          label="Gene"
          placeholder="Gene symbol, e.g. FBN1"
          options={annotations?.genes ?? []}
          value={colouring.term}
          loading={colouring.loading}
          onPick={colouring.setTerm}
          onClear={() => colouring.setTerm(null)}
        />
      )}

      {mode === 'symptom' && (
        <TermPicker
          label="Symptom or HPO term"
          placeholder="e.g. seizure, HP:0001250"
          options={annotations?.symptoms ?? []}
          value={colouring.term}
          loading={colouring.loading}
          onPick={colouring.setTerm}
          onClear={() => colouring.setTerm(null)}
        />
      )}

      {mode === 'disease' && (
        <TermPicker
          label="Disease"
          placeholder="Name or ORPHA ID"
          options={diseaseOptions}
          value={colouring.followsSelection ? null : colouring.term}
          countLabel="edges"
          onPick={colouring.setTerm}
          onClear={() => colouring.setTerm(null)}
        />
      )}
      {mode === 'disease' && colouring.followsSelection && (
        <p className="muted small">Following the disease you select. Pick one above to keep it fixed.</p>
      )}

      {(mode === 'onset' || mode === 'inheritance') && (
        <>
          {colouring.groups.length > 0 && (
            <p className="muted small">
              {mode === 'onset'
                ? 'Coloured by the earliest stage of onset listed for each disease.'
                : 'Diseases listing more than one mode are grouped together.'}{' '}
              Click a group to show every disease in it.
            </p>
          )}
          <ul className="cluster-list">
            {colouring.groups.map((g) => (
              <li key={g.key}>
                <button
                  type="button"
                  className="cluster-item"
                  aria-pressed={colouring.groupFocus === g.key}
                  onClick={() => colouring.setGroupFocus(colouring.groupFocus === g.key ? null : g.key)}
                >
                  <span className="swatch" style={{ background: g.colour }} aria-hidden />
                  <span className="cluster-name">{g.key}</span>
                  <span className="muted small">{g.count.toLocaleString()}</span>
                </button>
              </li>
            ))}
          </ul>
        </>
      )}

      {colouring.loading && <p className="muted small">Loading annotations…</p>}
      {colouring.error && <p className="callout small">{colouring.error}</p>}
      {colouring.summary && <p className="small colour-summary">{colouring.summary}</p>}
      {(mode === 'gene' || mode === 'symptom') && !colouring.term && !colouring.loading && !colouring.error && (
        <p className="muted small">Choose a {mode === 'gene' ? 'gene' : 'symptom'} to colour the diseases that have it.</p>
      )}

      <span className="section-label legend-subhead">Reading the graph</span>
      <ul>
        <li>
          <span className="line-key support-curated" aria-hidden /> Curated relation backs the edge
        </li>
        <li>
          <span className="line-key support-plausible" aria-hidden /> Plausible: shared group, gene or drug
        </li>
        <li>
          <span className="line-key support-novel" aria-hidden /> Novel: hypothesis to review
        </li>
        <li>
          <span className="line-key support-novel is-user" aria-hidden /> Dotted: from a disease you added
        </li>
        <li>
          <span className="node-key diamond" aria-hidden /> Added disease or paper (takes its neighbours' cluster)
        </li>
        <li>
          <span className="node-key sized" aria-hidden /> Larger node = more edges
        </li>
      </ul>
      <p className="muted small">
        Each disease links to its 10 most similar diseases under the v2 model. Nearby diseases in the layout have similar
        fused profiles. Hover a node for its ORPHA ID and cluster.
      </p>
    </div>
  )
}
