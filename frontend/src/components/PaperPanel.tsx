import { useEffect, useMemo, useState } from 'react'
import { type LoadedAnnotations, loadAnnotations } from '../data/annotations'
import type { ClaimVerdict, GraphNode, PaperEntry, PaperEvidenceItem } from '../data/types'
import { displayName, modalityLabel, plural, score } from '../lib/format'
import {
  type ClaimEvaluation,
  CARRY_MIN,
  countVerdicts,
  recall,
  VERDICT_LABEL,
  VERDICT_MEANING,
  VERDICT_ORDER,
} from '../lib/paperEval'
import { verdictColour } from '../lib/theme'
import type { UserDisease } from '../lib/userDiseases'

interface Props {
  paper: PaperEntry
  evaluations: ClaimEvaluation[]
  /** For an uploaded paper: its focal disease and placement. */
  disease: UserDisease | undefined
  nodes: Map<string, GraphNode>
  dark: boolean
  onSelectPair: (a: string, b: string) => void
  onSelectDisease: (id: string) => void
  onColourBy: (mode: 'gene' | 'symptom' | 'onset' | 'inheritance', term: string | null) => void
  onClose: () => void
}

const nameOf = (nodes: Map<string, GraphNode>, id: string) => displayName(nodes.get(id)?.name ?? id)

function VerdictPill({ verdict, dark }: { verdict: ClaimVerdict | undefined; dark: boolean }) {
  if (!verdict) return <span className="verdict-pill is-pending">Checking…</span>
  const colour = verdictColour(verdict, dark)
  return (
    <span className="verdict-pill" style={{ borderColor: colour, color: colour }} title={VERDICT_MEANING[verdict]}>
      <span className="verdict-dot" style={{ background: colour }} aria-hidden />
      {VERDICT_LABEL[verdict]}
    </span>
  )
}

/** The claims as a small diagram: the layout of the big graph packs related diseases on top of each other. */
function ClaimMap({
  evaluations,
  focalId,
  nodes,
  dark,
  onSelectPair,
  onSelectDisease,
}: {
  evaluations: ClaimEvaluation[]
  focalId: string | undefined
  nodes: Map<string, GraphNode>
  dark: boolean
} & Pick<Props, 'onSelectPair' | 'onSelectDisease'>) {
  const W = 420
  const H = 270
  const ids = [...new Set(evaluations.flatMap((e) => [e.claim.a, e.claim.b]))]
  const centre = focalId && ids.includes(focalId) ? focalId : null
  const ring = ids.filter((id) => id !== centre)
  const position = new Map<string, { x: number; y: number }>()
  if (centre) position.set(centre, { x: W / 2, y: H / 2 })
  ring.forEach((id, i) => {
    const angle = -Math.PI / 2 + (i / Math.max(ring.length, 1)) * 2 * Math.PI
    position.set(id, { x: W / 2 + Math.cos(angle) * 88, y: H / 2 + Math.sin(angle) * 92 })
  })
  const label = (id: string) => {
    const name = nameOf(nodes, id)
    return name.length > 20 ? `${name.slice(0, 19)}…` : name
  }
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="claim-map" role="img" aria-label="Diagram of the paper's claims, coloured by verdict">
      {evaluations.map((e, i) => {
        const a = position.get(e.claim.a)
        const b = position.get(e.claim.b)
        if (!a || !b) return null
        const colour = verdictColour(e.verdict ?? 'edge-only', dark)
        return (
          <line
            key={i}
            x1={a.x}
            y1={a.y}
            x2={b.x}
            y2={b.y}
            stroke={colour}
            strokeWidth={3}
            strokeDasharray={e.edge ? undefined : '6 5'}
            className={e.edge ? 'claim-map-edge' : undefined}
            onClick={() => (e.edge ? onSelectPair(e.claim.a, e.claim.b) : undefined)}
          >
            <title>{`${nameOf(nodes, e.claim.a)} – ${nameOf(nodes, e.claim.b)}: ${e.verdict ? VERDICT_LABEL[e.verdict] : 'checking'}`}</title>
          </line>
        )
      })}
      {ids.map((id) => {
        const at = position.get(id)!
        const right = at.x > W / 2 + 4
        const left = at.x < W / 2 - 4
        const focal = id === centre
        return (
          <g key={id} className="claim-map-node" onClick={() => onSelectDisease(id)}>
            <circle cx={at.x} cy={at.y} r={focal ? 8 : 6} className={focal ? 'is-focal' : undefined} />
            <text
              x={at.x + (right ? 11 : left ? -11 : 0)}
              y={at.y + (right || left ? 4 : at.y < H / 2 ? -12 : 20)}
              textAnchor={right ? 'start' : left ? 'end' : 'middle'}
            >
              {label(id)}
            </text>
            <title>{nameOf(nodes, id)}</title>
          </g>
        )
      })}
    </svg>
  )
}

function ClaimCard({
  evaluation,
  nodes,
  dark,
  onSelectPair,
  onSelectDisease,
}: { evaluation: ClaimEvaluation; nodes: Map<string, GraphNode>; dark: boolean } & Pick<Props, 'onSelectPair' | 'onSelectDisease'>) {
  const { claim, edge, dimensions, verdict } = evaluation
  return (
    <li className="claim-card">
      <div className="claim-head">
        <VerdictPill verdict={verdict} dark={dark} />
        <span className="muted small">
          {claim.relationship}
          {claim.stated !== null ? ` · stated ${claim.stated.toFixed(2)}` : ''}
        </span>
      </div>
      <p className="claim-pair">
        <button type="button" className="link-button" onClick={() => onSelectDisease(claim.a)}>
          {nameOf(nodes, claim.a)}
        </button>
        <span aria-hidden> – </span>
        <button type="button" className="link-button" onClick={() => onSelectDisease(claim.b)}>
          {nameOf(nodes, claim.b)}
        </button>
      </p>
      {claim.finding && (
        <p className="claim-finding small">
          “{claim.finding}”{claim.locator ? <span className="muted"> · {claim.locator}</span> : null}
        </p>
      )}

      <ul className="dimension-list">
        {dimensions.map((d) => (
          <li key={d.dimension} className={d.carried === null ? 'is-unchecked' : d.carried ? 'is-carried' : 'is-uncarried'}>
            <span className="dimension-mark" aria-hidden>
              {d.carried === null ? '•' : d.carried ? '✓' : '✗'}
            </span>
            <span className="dimension-body">
              <strong>{d.dimension}</strong>
              {d.carried === null ? (
                <span className="muted small"> no v2 modality, only the edge can be checked</span>
              ) : edge && !d.annotated ? (
                <span className="muted small"> not annotated for both diseases, so the graph had nothing to compare</span>
              ) : edge ? (
                <span className="muted small">
                  {' '}
                  {d.contributions
                    .map((c) => `${modalityLabel(c.modality)} ${c.contribution > 0 ? '+' : ''}${score(c.contribution)}`)
                    .join(' · ')}
                </span>
              ) : (
                <span className="muted small"> no edge to read it from</span>
              )}
            </span>
          </li>
        ))}
      </ul>
      {claim.feature && evaluation.featureShared !== null && (
        <p className="small muted">
          Named feature “{claim.feature}”: {evaluation.featureShared ? 'listed among the edge’s shared features' : 'not among the edge’s top shared features'}.
        </p>
      )}

      <div className="claim-actions">
        {edge ? (
          <button type="button" className="link-button small" onClick={() => onSelectPair(claim.a, claim.b)}>
            View edge explanation
          </button>
        ) : (
          <span className="muted small">Not among each other’s most similar diseases.</span>
        )}
      </div>
    </li>
  )
}

const FEATURE_MODE: Record<string, 'gene' | 'symptom' | 'onset' | 'inheritance' | undefined> = {
  gene: 'gene',
  phenotype: 'symptom',
  onset: 'onset',
  inheritance: 'inheritance',
}

/** How many of the closest diseases carry a feature the paper reports for its focal disease. */
function Coverage({
  items,
  disease,
  onColourBy,
}: { items: PaperEvidenceItem[]; disease: UserDisease | undefined } & Pick<Props, 'onColourBy'>) {
  const [ann, setAnn] = useState<LoadedAnnotations | null>(null)
  const [failed, setFailed] = useState(false)
  useEffect(() => {
    let cancelled = false
    loadAnnotations().then(
      (a) => !cancelled && setAnn(a),
      () => !cancelled && setFailed(true),
    )
    return () => {
      cancelled = true
    }
  }, [])

  const neighbours = useMemo(() => (disease?.placement?.neighbours ?? []).slice(0, 10), [disease])
  const rows = useMemo(() => {
    if (!ann) return []
    return items.map((item) => {
      const lists: number[][] | null =
        item.feature_type === 'gene'
          ? [ann.raw.genes[item.identifier] ?? []]
          : item.feature_type === 'phenotype'
            ? [ann.raw.phenotypes[item.identifier]?.n ?? []]
            : item.feature_type === 'onset'
              ? [ann.raw.onset[item.identifier] ?? []]
              : item.feature_type === 'inheritance'
                ? [ann.raw.inheritance[item.identifier] ?? []]
                : null
      if (!lists) return { item, carriers: null as string[] | null }
      const set = new Set(lists[0])
      const carriers = neighbours.filter((n) => {
        const i = ann.index.get(n.id)
        return i !== undefined && set.has(i)
      })
      return { item, carriers: carriers.map((n) => n.name) }
    })
  }, [ann, items, neighbours])

  if (failed) return <p className="muted small">Annotations are not built, so the profile cannot be checked against the graph.</p>
  if (!ann) return <p className="muted small">Loading annotations…</p>
  const checked = rows.filter((r) => r.carriers !== null)
  const unreflected = checked.filter((r) => r.carriers!.length === 0)
  return (
    <>
      <p className="small">
        <strong>{checked.length - unreflected.length}</strong> of {plural(checked.length, 'checkable feature')} the paper reports
        for its disease {checked.length - unreflected.length === 1 ? 'is' : 'are'} also carried by at least one of its{' '}
        {neighbours.length} closest diseases in the graph.
      </p>
      <ul className="coverage-list">
        {rows.map(({ item, carriers }) => {
          const mode = FEATURE_MODE[item.feature_type]
          return (
            <li key={`${item.feature_type}:${item.identifier}`}>
              <div className="coverage-head">
                <span className="chip">{item.feature_type}</span>
                <strong>{item.label}</strong>
                {item.label !== item.identifier && <span className="muted small">{item.identifier}</span>}
                <span className="muted small coverage-count">
                  {carriers === null ? 'not checked' : `${carriers.length}/${neighbours.length} neighbours`}
                </span>
              </div>
              {carriers !== null && (
                <div className="bar-track" title={carriers.map(displayName).join(', ') || 'none'}>
                  <span
                    className={`bar-fill ${carriers.length === 0 ? 'is-empty' : ''}`}
                    style={{ width: `${Math.max((carriers.length / Math.max(neighbours.length, 1)) * 100, carriers.length ? 4 : 0)}%` }}
                  />
                </div>
              )}
              <p className="claim-finding small muted">
                “{item.quote}” · {item.locator}
                {item.verification_status !== 'unverified' ? ` · ${item.verification_status}` : ''}
              </p>
              {mode && (
                <button type="button" className="link-button small" onClick={() => onColourBy(mode, item.identifier)}>
                  Colour the graph by this {item.feature_type === 'phenotype' ? 'symptom' : item.feature_type}
                </button>
              )}
            </li>
          )
        })}
      </ul>
      {unreflected.length > 0 && (
        <p className="muted small">
          Features carried by no neighbour are what the paper reports but the graph’s nearest diseases do not share; they are
          either specific to this paper or evidence the model is not using.
        </p>
      )}
    </>
  )
}

export function PaperPanel(props: Props) {
  const { paper, evaluations, disease, nodes, dark } = props
  const [filter, setFilter] = useState<ClaimVerdict | null>(null)
  const counts = useMemo(() => countVerdicts(evaluations), [evaluations])
  const judged = VERDICT_ORDER.reduce((s, v) => s + counts[v], 0)
  const checkable = counts.confirmed + counts.partial + counts.unsupported
  const shown = filter ? evaluations.filter((e) => e.verdict === filter) : evaluations
  const result = paper.result

  return (
    <div className="paper-panel panel-body">
      <section className="panel-section">
        <p className="eyebrow">{paper.kind === 'upload' ? 'Uploaded paper' : 'Literature paper'}</p>
        <h2>{paper.title}</h2>
        <p className="muted small">
          {paper.year ?? ''}
          {paper.link && (
            <>
              {paper.year ? ' · ' : ''}
              <a href={paper.link} target="_blank" rel="noreferrer">
                Open paper
              </a>
            </>
          )}
        </p>
        {paper.mock && (
          <p className="callout small">
            <strong>Mock extraction.</strong> The profile, quotes and comparisons below are generated from the graph, not read
            from the paper. They show what the v2_5 output will look like once it is connected.
          </p>
        )}
        {result?.warnings.filter((w) => !paper.mock || !w.startsWith('Mock output')).map((w) => (
          <p key={w} className="callout small">
            {w}
          </p>
        ))}
      </section>

      <section className="panel-section">
        <h3>Does the graph carry the paper’s evidence?</h3>
        {evaluations.length === 0 ? (
          <p className="muted small">
            This paper states no comparison between two diseases, so there is no pair to check. The extracted profile below is
            checked instead.
          </p>
        ) : (
          <>
            <p className="small">
              <strong>{Math.round(recall(counts) * 100)}%</strong> of {plural(judged, 'claim')} have an edge in the graph
              {checkable > 0 && (
                <>
                  ; <strong>{counts.confirmed}</strong> of {checkable} with a checkable dimension have every named dimension
                  contributing to the edge score (≥ {CARRY_MIN} logit).
                </>
              )}
            </p>
            <div className="verdict-bar" role="img" aria-label="Claims by verdict">
              {VERDICT_ORDER.map((v) =>
                counts[v] > 0 ? <span key={v} style={{ flex: counts[v], background: verdictColour(v, dark) }} /> : null,
              )}
            </div>
            <ul className="verdict-legend">
              {VERDICT_ORDER.map((v) => (
                <li key={v}>
                  <button
                    type="button"
                    className="verdict-filter"
                    aria-pressed={filter === v}
                    disabled={counts[v] === 0}
                    title={VERDICT_MEANING[v]}
                    onClick={() => setFilter(filter === v ? null : v)}
                  >
                    <span className="verdict-dot" style={{ background: verdictColour(v, dark) }} aria-hidden />
                    {VERDICT_LABEL[v]} <span className="muted">{counts[v]}</span>
                  </button>
                </li>
              ))}
            </ul>
            <p className="muted small">
              Graph edges are drawn in these colours; dashed grey lines mark claims the graph has no edge for. Click a colour to
              list only those claims.
            </p>
          </>
        )}
      </section>

      {evaluations.length > 0 && (
        <section className="panel-section">
          <h3>
            Claims <span className="muted">{shown.length}</span>
          </h3>
          <ClaimMap
            evaluations={shown}
            focalId={paper.focalId}
            nodes={nodes}
            dark={dark}
            onSelectPair={props.onSelectPair}
            onSelectDisease={props.onSelectDisease}
          />
          <ul className="claim-list">
            {shown.map((e, i) => (
              <ClaimCard
                key={`${e.claim.a}|${e.claim.b}|${i}`}
                evaluation={e}
                nodes={nodes}
                dark={dark}
                onSelectPair={props.onSelectPair}
                onSelectDisease={props.onSelectDisease}
              />
            ))}
          </ul>
        </section>
      )}

      {result && result.accepted_evidence.length > 0 && (
        <section className="panel-section">
          <h3>Extracted profile, checked against the graph</h3>
          <Coverage items={result.accepted_evidence} disease={disease} onColourBy={props.onColourBy} />
          {result.rejected_features.length > 0 && (
            <p className="muted small">{plural(result.rejected_features.length, 'extracted feature')} rejected during validation.</p>
          )}
        </section>
      )}

      <button type="button" className="button full" onClick={props.onClose}>
        Close paper
      </button>
    </div>
  )
}
