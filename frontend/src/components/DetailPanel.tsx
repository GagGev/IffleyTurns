import { useEffect, useMemo, useState } from 'react'
import { loadEdgeDetail } from '../data/source'
import type { ApiHealth, EdgeDetail, GraphData, GraphEdge, GraphNode } from '../data/types'
import { MODALITY_LABELS, SUPPORT_MEANING } from '../data/types'
import { findEdge, otherEnd } from '../lib/graph'
import {
  capitalise,
  displayName,
  download,
  isOrpha,
  modalityLabel,
  orphanetUrl,
  plural,
  score,
} from '../lib/format'
import type { Selection } from '../lib/selection'
import { FREQUENCIES, phenotypeWeights, type UserDisease } from '../lib/userDiseases'
import { EdgeExplanation } from './EdgeExplanation'
import type { Literature } from '../data/annotations'
import { DiseasePapers } from './DiseasePapers'

/** What the panel needs to show and act on the user's added diseases. */
export interface UserContext {
  diseases: Map<string, UserDisease>
  api: ApiHealth
  placing: Set<string>
  errors: Map<string, string>
  /** Explanations for edges from added diseases, taken from their placements. */
  edgeDetails: Map<string, EdgeDetail>
  onEdit: (disease: UserDisease) => void
  onRemove: (disease: UserDisease) => void
  onPlace: (id: string) => void
}

interface Props {
  graph: GraphData
  nodes: Map<string, GraphNode>
  edgesById: Map<string, GraphEdge>
  edgesByNode: Map<string, GraphEdge[]>
  visible: GraphEdge[]
  visibleIds: Set<string>
  selection: Selection
  user: UserContext
  clusterColour: (index: number) => string
  focusCluster: number | null
  onFocusCluster: (next: number | null) => void
  onSelectDisease: (id: string) => void
  onSelectPair: (a: string, b: string) => void
  /** The literature set, for listing a disease's papers. */
  literature: Literature | null
  literatureError: string | null
  /** Show a paper's claims on the graph. */
  onOpenPaper: (paperId: string) => void
}

export function DetailPanel(props: Props) {
  const { selection } = props
  if (!selection) return null
  if (selection.kind === 'disease') {
    const added = props.user.diseases.get(selection.id)
    if (added) return <UserDiseaseDetail {...props} disease={added} />
    return <DiseaseDetail {...props} id={selection.id} />
  }
  return <PairDetail {...props} a={selection.a} b={selection.b} />
}

function ClusterCard({
  cluster,
  colour,
  active,
  isHub,
  onToggle,
}: {
  cluster: GraphData['clusters'][number]
  colour: string
  active: boolean
  isHub: boolean
  onToggle: () => void
}) {
  return (
    <div className="cluster-card">
      <span className="swatch" style={{ background: colour }} aria-hidden />
      <div>
        <strong>
          {cluster.id} · {cluster.label}
        </strong>
        <p className="muted small">
          {plural(cluster.size, 'disease')} · {Math.round(cluster.purity * 100)}% {cluster.topCategory}
          {isHub ? ' · hub of this cluster' : ` · hub: ${displayName(cluster.hubName)}`}
        </p>
        <button type="button" className="link-button" onClick={onToggle}>
          {active ? 'Stop highlighting cluster' : 'Highlight cluster in graph'}
        </button>
      </div>
    </div>
  )
}

function SupportKey({ edge }: { edge: GraphEdge }) {
  return <span className={`line-key support-${edge.support}${edge.origin === 'user' ? ' is-user' : ''}`} aria-hidden />
}

/** Column names for a pair-list; the rows share its grid, so the columns line up. */
function PairListHead() {
  return (
    <li className="pair-list-head" aria-hidden>
      <span />
      <span>Disease</span>
      <span className="num">Score</span>
      <span>Main evidence</span>
    </li>
  )
}

function EdgeRow({ edge, other, nodes, filtered, onClick }: {
  edge: GraphEdge
  other: string
  nodes: Map<string, GraphNode>
  filtered?: boolean
  onClick: () => void
}) {
  return (
    <li className={filtered ? 'is-filtered' : undefined}>
      <button type="button" onClick={onClick}>
        <SupportKey edge={edge} />
        <span className="pair-names">{displayName(nodes.get(other)?.name ?? other)}</span>
        <span className="num">{score(edge.score)}</span>
        <span className="muted">{modalityLabel(edge.mainModality)}</span>
      </button>
    </li>
  )
}

// --- Disease -----------------------------------------------------------------------

function ModalityChips({ graph, present }: { graph: GraphData; present: string[] }) {
  return (
    <>
      <p className="muted small">
        {present.length} of {graph.modalities.length} modalities annotated
      </p>
      <ul className="modality-chips">
        {graph.modalities.map((m) => (
          <li
            key={m}
            className={present.includes(m) ? 'is-present' : undefined}
            title={graph.modalityDescriptions[m]}
          >
            {modalityLabel(m)}
          </li>
        ))}
      </ul>
    </>
  )
}

function DiseaseDetail({
  id,
  graph,
  nodes,
  edgesByNode,
  visibleIds,
  onSelectPair,
  clusterColour,
  focusCluster,
  onFocusCluster,
  literature,
  literatureError,
  onOpenPaper,
}: Props & { id: string }) {
  const node = nodes.get(id)
  const incident = useMemo(() => [...(edgesByNode.get(id) ?? [])].sort((a, b) => b.score - a.score), [edgesByNode, id])
  if (!node) {
    return (
      <div className="panel-body">
        <p className="muted panel-section">“{id}” is not in the graph.</p>
      </div>
    )
  }
  const hidden = incident.filter((e) => !visibleIds.has(e.id)).length
  return (
    <div className="panel-body">
      <section className="panel-section">
        <p className="eyebrow">{node.origin === 'shared' ? 'Added with place_disease.py' : 'Disease'}</p>
        <h2>{displayName(node.name)}</h2>
        <p className="meta-row">
          {isOrpha(node.id) && (
            <a href={orphanetUrl(node.id)} target="_blank" rel="noreferrer">
              {node.id}
            </a>
          )}
          {node.category && <span className="status">{node.category}</span>}
        </p>
        {node.disorderType && <p className="muted small">{node.disorderType}</p>}
        {graph.clusters[node.cluster] && (
          <ClusterCard
            cluster={graph.clusters[node.cluster]}
            colour={clusterColour(node.cluster)}
            active={focusCluster === node.cluster}
            onToggle={() => onFocusCluster(focusCluster === node.cluster ? null : node.cluster)}
            isHub={graph.clusters[node.cluster].hubId === node.id}
          />
        )}
      </section>
      <section className="panel-section">
        <h3>Profile</h3>
        <ModalityChips graph={graph} present={node.modalities} />
      </section>
      <section className="panel-section">
        <h3>
          Most similar diseases <span className="muted">· {plural(incident.length, 'edge')}</span>
        </h3>
        <p className="muted small">
          Ranked by similarity score; higher is more similar.
          {hidden > 0 && ` ${hidden} hidden from the graph by the current filters.`}
        </p>
        <ul className="pair-list">
          <PairListHead />
          {incident.map((e) => (
            <EdgeRow
              key={e.id}
              edge={e}
              other={otherEnd(e, id)}
              nodes={nodes}
              filtered={!visibleIds.has(e.id)}
              onClick={() => onSelectPair(id, otherEnd(e, id))}
            />
          ))}
        </ul>
      </section>
      <DiseasePapers
        key={id}
        id={id}
        literature={literature}
        error={literatureError}
        nodes={nodes}
        onSelectPair={onSelectPair}
        onOpenPaper={onOpenPaper}
      />
    </div>
  )
}

// --- Added disease -----------------------------------------------------------------

function placementBlocker(api: ApiHealth): string | null {
  if (api.status === 'ready') return null
  if (api.status === 'loading') return 'The v2 model is still loading.'
  if (api.status === 'error') return `The placement service could not load the v2 model: ${api.error}`
  return 'The placement service is not running. Start it with `python frontend/api/server.py`.'
}

function UserDiseaseDetail({ disease, graph, nodes, edgesByNode, user, onSelectPair }: Props & { disease: UserDisease }) {
  const { input, placement, labels = {} } = disease
  const incident = useMemo(
    () => [...(edgesByNode.get(disease.id) ?? [])].sort((a, b) => b.score - a.score),
    [edgesByNode, disease.id],
  )
  const blocker = placementBlocker(user.api)
  const placing = user.placing.has(disease.id)
  const error = user.errors.get(disease.id)
  const phenotypes = Object.entries(phenotypeWeights(input))
  const frequency = (w: number) => FREQUENCIES.find((f) => Math.abs(f.weight - w) < 0.01)?.label ?? w.toFixed(2)
  const list = (values?: string[]) =>
    values?.length ? (
      <span className="shared-terms">
        {values.map((v) => (
          <span key={v} className="chip" title={labels[v] ?? v}>
            {v}
            {labels[v] && <span className="term-name"> {labels[v]}</span>}
          </span>
        ))}
      </span>
    ) : (
      <span className="no-evidence">Not given</span>
    )

  return (
    <div className="panel-body">
      <section className="panel-section">
        <p className="eyebrow">
          <span className="user-mark" aria-hidden>
            ◆
          </span>{' '}
          Added by you
        </p>
        <h2>{input.name}</h2>
        <p className="meta-row">
          <button type="button" className="link-button" onClick={() => user.onEdit(disease)}>
            Edit
          </button>
          <button
            type="button"
            className="link-button"
            disabled={!!blocker || placing}
            title={blocker ?? undefined}
            onClick={() => user.onPlace(disease.id)}
          >
            {placing ? 'Placing…' : placement ? 'Place again' : 'Place now'}
          </button>
          <button
            type="button"
            className="link-button"
            onClick={() => download(`${disease.id.replace(/[^a-z0-9-]+/gi, '_')}.json`, JSON.stringify({ id: disease.id, ...input }, null, 2), 'application/json')}
          >
            Download v2 JSON
          </button>
          <button type="button" className="link-button" onClick={() => user.onRemove(disease)}>
            Remove
          </button>
        </p>
        {error && <p className="callout">{error}</p>}
        {!placement && !error && (
          <p className="callout">{blocker ? `Not placed yet. ${blocker}` : 'Not placed yet.'}</p>
        )}
      </section>

      {placement && (
        <>
          <section className="panel-section">
            <h3>
              Most similar diseases <span className="muted">· top {incident.length}</span>
            </h3>
            <p className="muted small">
              Placed {new Date(placement.placedAt).toLocaleString()} using{' '}
              {placement.present.map((m) => MODALITY_LABELS[m] ?? m).join(', ') || 'no modalities'}.
            </p>
            <ul className="pair-list">
              <PairListHead />
              {incident.map((e) => (
                <EdgeRow
                  key={e.id}
                  edge={e}
                  other={otherEnd(e, disease.id)}
                  nodes={nodes}
                  onClick={() => onSelectPair(disease.id, otherEnd(e, disease.id))}
                />
              ))}
            </ul>
          </section>
          {placement.warnings.length > 0 && (
            <section className="panel-section">
              <h3>Warnings from v2</h3>
              <ul className="warnings">
                {placement.warnings.map((w) => (
                  <li key={w}>{w}</li>
                ))}
              </ul>
            </section>
          )}
          <section className="panel-section">
            <details>
              <summary>Model weights used for this disease</summary>
              <p className="muted small">
                v2 refits its fusion to the modalities a new disease has, moving weight onto those it can use.
              </p>
              <table className="components">
                <tbody>
                  {Object.entries(placement.weights)
                    .sort((a, b) => b[1] - a[1])
                    .map(([m, w]) => (
                      <tr key={m}>
                        <th scope="row" title={graph.modalityDescriptions[m]}>
                          {modalityLabel(m)}
                        </th>
                        <td className="num">{score(w)}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </details>
          </section>
        </>
      )}

      <section className="panel-section">
        <h3>Description entered</h3>
        <dl className="feature-list">
          <div>
            <dt>Description</dt>
            <dd>{input.description?.trim() || <span className="no-evidence">Not given</span>}</dd>
          </div>
          <div>
            <dt>Synonyms</dt>
            <dd>{list(input.synonyms)}</dd>
          </div>
          <div>
            <dt>Phenotypes</dt>
            <dd>
              {phenotypes.length === 0 ? (
                <span className="no-evidence">Not given</span>
              ) : (
                <span className="shared-terms">
                  {phenotypes.map(([term, w]) => (
                    <span key={term} className="chip" title={labels[term] ?? term}>
                      {labels[term] ?? term} <span className="term-name">· {frequency(w)}</span>
                    </span>
                  ))}
                </span>
              )}
            </dd>
          </div>
          <div>
            <dt>Genes</dt>
            <dd>{list(input.genes)}</dd>
          </div>
          <div>
            <dt>Drugs</dt>
            <dd>{list(input.drugs)}</dd>
          </div>
          <div>
            <dt>Inheritance</dt>
            <dd>{list(input.inheritance)}</dd>
          </div>
          <div>
            <dt>Onset</dt>
            <dd>{list(input.onset)}</dd>
          </div>
          <div>
            <dt>Prevalence</dt>
            <dd>
              {input.prevalence ? (
                `1 in ${Math.round(1 / input.prevalence).toLocaleString()}`
              ) : (
                <span className="no-evidence">Not given</span>
              )}
            </dd>
          </div>
          <div>
            <dt>Classification</dt>
            <dd>{list(input.ontology_parents)}</dd>
          </div>
        </dl>
      </section>
    </div>
  )
}

const scoreRanges = new WeakMap<GraphData, { median: number; max: number }>()

/** "Higher than most links in the graph (median 2.5)": puts a raw model score in context. */
function scoreSentence(value: number, graph: GraphData): string {
  let range = scoreRanges.get(graph)
  if (!range) {
    const scores = graph.edges.map((e) => e.score).sort((a, b) => a - b)
    range = { median: scores[Math.floor(scores.length / 2)] ?? 0, max: scores[scores.length - 1] ?? 0 }
    scoreRanges.set(graph, range)
  }
  const relation = value >= range.median ? 'above' : 'below'
  return `Higher means more similar. This is ${relation} the median link in the graph (${score(range.median)}; the highest is ${score(range.max)}).`
}

// --- Pair --------------------------------------------------------------------------

function useEdgeDetail(graph: GraphData, edge: GraphEdge | undefined, userDetails: Map<string, EdgeDetail>) {
  const [state, setState] = useState<{ id: string; detail?: EdgeDetail | null; error?: string }>()
  const userDetail = edge?.origin === 'user' ? userDetails.get(edge.id) : undefined
  useEffect(() => {
    if (!edge || edge.origin === 'user') return
    let cancelled = false
    loadEdgeDetail(graph, edge.id).then(
      (detail) => !cancelled && setState({ id: edge.id, detail }),
      (e: Error) => !cancelled && setState({ id: edge.id, error: e.message }),
    )
    return () => {
      cancelled = true
    }
  }, [graph, edge])
  if (!edge) return { detail: null }
  if (edge.origin === 'user') return { detail: userDetail ?? null }
  if (state?.id !== edge.id) return { loading: true, detail: null }
  return { detail: state.detail ?? null, error: state.error }
}

function PairDetail({ a, b, graph, nodes, edgesById, user, onSelectDisease }: Props & { a: string; b: string }) {
  const edge = findEdge(edgesById, a, b)
  const { detail, loading, error } = useEdgeDetail(graph, edge, user.edgeDetails)
  const na = nodes.get(edge?.source ?? a)
  const nb = nodes.get(edge?.target ?? b)

  const nameButton = (n: GraphNode | undefined, fallback: string) => (
    <button type="button" className="link-button" onClick={() => onSelectDisease(n?.id ?? fallback)}>
      {n?.origin === 'user' && (
        <span className="user-mark" aria-hidden>
          ◆{' '}
        </span>
      )}
      {displayName(n?.name ?? fallback)}
    </button>
  )

  return (
    <div className="panel-body">
      <section className="panel-section">
        <p className="eyebrow">Disease pair</p>
        <h2 className="pair-heading">
          {nameButton(na, a)}
          <span className="pair-sep" aria-hidden>
            ↔
          </span>
          {nameButton(nb, b)}
        </h2>
        <p className="meta-row small">
          {[na, nb].map((n) =>
            n && isOrpha(n.id) ? (
              <a key={n.id} href={orphanetUrl(n.id)} target="_blank" rel="noreferrer">
                {n.id}
              </a>
            ) : null,
          )}
        </p>
      </section>

      {!edge ? (
        <section className="panel-section">
          <p className="muted">
            These diseases are not among each other's top {graph.k ?? 10} neighbours, so the graph has no edge between
            them.
          </p>
        </section>
      ) : (
        <>
          <section className="panel-section">
            <div className="stats">
              <div className="stat">
                <span className="stat-value">{score(edge.score)}</span>
                <span className="stat-label">Score</span>
              </div>
              <div className="stat">
                <span className="stat-value stat-support">
                  <SupportKey edge={edge} />
                  {capitalise(edge.support)}
                </span>
                <span className="stat-label">Support</span>
              </div>
            </div>
            <p className="small">{scoreSentence(edge.score, graph)}</p>
            <p className="muted small">{SUPPORT_MEANING[edge.support]}</p>
            {edge.origin === 'graph' && (
              <p className="muted small">
                {edge.mutual
                  ? 'Mutual: each lists the other among its top neighbours.'
                  : 'One-way: only one of the two lists the other among its top neighbours.'}{' '}
                {[
                  detail?.ranks[0] && `#${detail.ranks[0]} among ${displayName(na?.name ?? a)}'s neighbours`,
                  detail?.ranks[1] && `#${detail.ranks[1]} among ${displayName(nb?.name ?? b)}'s neighbours`,
                ]
                  .filter(Boolean)
                  .join('; ')
                  .replace(/^./, (c) => c.toUpperCase())}
                {detail?.ranks.some(Boolean) ? '.' : ''}
              </p>
            )}
          </section>
          {loading && <p className="muted panel-section">Loading explanation…</p>}
          {error && <p className="callout panel-section">{error}</p>}
          {detail && <EdgeExplanation detail={detail} modalities={graph.modalities} descriptions={graph.modalityDescriptions} />}
          {!loading && !error && !detail && (
            <p className="muted panel-section">No explanation was recorded for this edge.</p>
          )}
        </>
      )}
    </div>
  )
}
