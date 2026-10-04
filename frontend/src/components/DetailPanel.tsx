import { useMemo } from 'react'
import { type FeatureCatalogue, type FeatureFamily, type FeatureProfile, FAMILY_COLUMNS, SET_FAMILIES } from '../data/features'
import type { Disease, Edge, LiteratureGraph } from '../data/types'
import { type EdgeView, type Filters, otherEnd, pairId, viewEdge } from '../lib/graph'
import { capitalise, displayName, measureLabel, orphanetUrl, plural, score } from '../lib/format'
import type { Selection } from '../lib/selection'
import { computeSimilarity, rankNeighbours, type Weights } from '../lib/similarity'
import type { FamilyColumn, UserDisease } from '../lib/userDiseases'
import { ComputedSimilarityCard } from './ComputedSimilarityCard'
import { DimensionBars } from './DimensionBars'
import { NeighbourList } from './NeighbourList'
import { PaperList } from './PaperList'

/** Everything the panel needs for feature-based similarity. */
export interface FeatureContext {
  /** undefined while loading, null when no profiles have been generated. */
  catalogue: FeatureCatalogue | null | undefined
  /** Catalogue profiles plus the user's diseases. */
  profiles: FeatureProfile[]
  profileFor: (diseaseId: string) => FeatureProfile | undefined
  weights: Weights
  userDiseases: Map<string, UserDisease>
  onEditUser: (disease: UserDisease) => void
  onRemoveUser: (disease: UserDisease) => void
}

interface Props {
  graph: LiteratureGraph
  diseases: Map<string, Disease>
  edgesById: Map<string, Edge>
  edgesByDisease: Map<string, Edge[]>
  visible: EdgeView[]
  visibleIds: Set<string>
  filters: Filters
  selection: Selection
  features: FeatureContext
  onSelectDisease: (id: string) => void
  onSelectPair: (a: string, b: string) => void
}

const FAMILY_TITLES: Record<FeatureFamily, string> = {
  phenotypes: 'Phenotypes',
  genes: 'Genes',
  classifications: 'Categories',
  body_systems: 'Body systems',
  inheritance: 'Inheritance',
  onset: 'Onset',
  approved_drugs: 'Approved drugs',
}

export function DetailPanel(props: Props) {
  const { selection } = props
  if (!selection) return <Overview {...props} />
  if (selection.kind === 'disease') {
    const user = props.features.userDiseases.get(selection.id)
    if (user) return <UserDiseaseDetail {...props} disease={user} />
    return <DiseaseDetail {...props} id={selection.id} />
  }
  return <PairDetail {...props} a={selection.a} b={selection.b} />
}

function OrphaLinks({ disease }: { disease: Disease }) {
  if (disease.orphaId) {
    return (
      <a href={orphanetUrl(disease.orphaId)} target="_blank" rel="noreferrer">
        {disease.orphaId}
      </a>
    )
  }
  if (disease.origin === 'user') return null
  return <span className="warning-text">No single ORPHA ID</span>
}

/** Why there is no feature profile for a disease, in words a researcher can act on. */
function missingProfileReason(disease: Disease | undefined, features: FeatureContext): string {
  if (features.catalogue === undefined) return 'Loading feature profiles…'
  if (features.catalogue === null) {
    return 'Feature profiles for catalogue diseases have not been generated yet. Run generate_features.py, then npm run features.'
  }
  if (!disease) return 'This disease has no feature profile.'
  if (!disease.orphaId) return `${displayName(disease.name)} has no single ORPHA ID, so it has no feature profile.`
  return `${disease.orphaId} is not in the feature catalogue.`
}

function Overview({ graph, visible, diseases, onSelectPair }: Props) {
  const strongest = useMemo(
    () => [...visible].sort((x, y) => y.agg.evidenceScore - x.agg.evidenceScore || y.value - x.value).slice(0, 8),
    [visible],
  )
  return (
    <div className="panel-body">
      <section className="panel-section">
        <h2>Explore literature-backed relationships</h2>
        <p>
          Each line joins two diseases that published papers compare. Select a disease or a line to see the evidence:
          the papers, how strong their study designs are, and which similarity dimensions they support.
        </p>
        <p>
          Use <strong>Add a disease</strong> to place your own disease in the graph from a form or a JSON file and see
          which diseases it most resembles.
        </p>
        <p className="muted small">
          Built from {plural(graph.stats.paperRows, 'paper row')} across {graph.datasets.length} curated sources. Scores
          are abstract-level judgements, not validated measurements.
        </p>
      </section>
      <section className="panel-section">
        <h3>Strongest evidence in view</h3>
        {strongest.length === 0 ? (
          <p className="muted">No pairs match the current filters.</p>
        ) : (
          <ul className="pair-list">
            {strongest.map(({ edge, agg, value }) => (
              <li key={edge.id}>
                <button type="button" onClick={() => onSelectPair(edge.source, edge.target)}>
                  <span className={`line-key rel-${agg.relationship.replace(/ /g, '-')}`} aria-hidden />
                  <span className="pair-names">
                    {displayName(diseases.get(edge.source)?.name ?? edge.source)} –{' '}
                    {displayName(diseases.get(edge.target)?.name ?? edge.target)}
                  </span>
                  <span className="num">{score(value)}</span>
                  <span className="muted">{agg.evidenceScore}/15</span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  )
}

function FeatureNeighbours({
  id,
  disease,
  features,
  limit,
  onSelectPair,
}: {
  id: string
  disease: Disease | undefined
  features: FeatureContext
  limit: number
  onSelectPair: (a: string, b: string) => void
}) {
  const profile = features.profileFor(id)
  const neighbours = useMemo(
    () => (profile ? rankNeighbours(profile, features.profiles, limit, features.weights) : []),
    [profile, features.profiles, features.weights, limit],
  )
  if (!profile) return <p className="muted">{missingProfileReason(disease, features)}</p>
  return (
    <NeighbourList
      target={profile}
      neighbours={neighbours}
      weights={features.weights}
      onSelect={(other) => onSelectPair(id, other)}
      empty={
        features.profiles.length <= 1
          ? 'There is nothing to compare with yet.'
          : 'No other disease shares any of these features.'
      }
    />
  )
}

function DiseaseDetail(props: Props & { id: string }) {
  const { id, diseases, edgesByDisease, filters, visibleIds, features, onSelectPair } = props
  const profile = features.profileFor(id)
  const disease: Disease | undefined =
    diseases.get(id) ??
    (profile && {
      id,
      name: profile.name,
      aliases: [],
      orphaId: id,
      candidateOrphaIds: [],
      status: 'unknown',
      degree: 0,
      origin: 'catalogue',
    })

  const related = useMemo(() => {
    const views = (edgesByDisease.get(id) ?? [])
      .map((e) => viewEdge(e, filters))
      .filter((v): v is EdgeView => v !== null)
    return views.sort((x, y) => y.value - x.value || y.agg.evidenceScore - x.agg.evidenceScore)
  }, [id, edgesByDisease, filters])

  if (!disease) {
    return (
      <div className="panel-body">
        <p className="muted panel-section">“{id}” is not in the literature graph or the feature catalogue.</p>
      </div>
    )
  }

  const totalPapers = related.reduce((s, v) => s + v.agg.nPapers, 0)
  const hidden = related.filter((v) => !visibleIds.has(v.edge.id)).length

  return (
    <div className="panel-body">
      <section className="panel-section">
        <p className="eyebrow">{disease.origin === 'catalogue' ? 'Catalogue disease' : 'Disease'}</p>
        <h2>{displayName(disease.name)}</h2>
        <p className="meta-row">
          <OrphaLinks disease={disease} />
          {disease.status !== 'unknown' && (
            <span className={`status status-${disease.status}`}>{capitalise(disease.status)}</span>
          )}
        </p>
        {!disease.orphaId && (
          <p className="callout">
            The literature names this disease without a single Orphanet match, so feature-based comparison is not
            possible yet.
            {disease.candidateOrphaIds.length > 0 && (
              <>
                {' '}
                Candidate IDs:{' '}
                {disease.candidateOrphaIds.map((c, i) => (
                  <span key={c}>
                    {i > 0 && ', '}
                    <a href={orphanetUrl(c)} target="_blank" rel="noreferrer">
                      {c}
                    </a>
                  </span>
                ))}
              </>
            )}
          </p>
        )}
        {disease.aliases.length > 0 && <p className="muted small">Also recorded as: {disease.aliases.join('; ')}</p>}
      </section>

      <section className="panel-section">
        <h3>
          Related in the literature{' '}
          <span className="muted">
            · {plural(related.length, 'pair')}, {plural(totalPapers, 'paper')}
          </span>
        </h3>
        {related.length > 0 && (
          <p className="muted small">
            Ranked by {measureLabel(filters.measure).toLowerCase()}.
            {hidden > 0 && ` ${hidden} hidden from the graph by the current filters.`}
          </p>
        )}
        {related.length === 0 ? (
          <p className="muted">No literature pairs for the selected sources and score.</p>
        ) : (
          <ul className="pair-list">
            {related.map(({ edge, agg, value }) => {
              const other = otherEnd(edge, id)
              return (
                <li key={edge.id} className={visibleIds.has(edge.id) ? undefined : 'is-filtered'}>
                  <button type="button" onClick={() => onSelectPair(id, other)}>
                    <span className={`line-key rel-${agg.relationship.replace(/ /g, '-')}`} aria-hidden />
                    <span className="pair-names">{displayName(diseases.get(other)?.name ?? other)}</span>
                    <span className="num">{score(value)}</span>
                    <span className="muted">{plural(agg.nPapers, 'paper')}</span>
                  </button>
                </li>
              )
            })}
          </ul>
        )}
      </section>

      <section className="panel-section">
        <h3>Feature-based nearest neighbours</h3>
        <FeatureNeighbours id={id} disease={disease} features={features} limit={10} onSelectPair={onSelectPair} />
      </section>
    </div>
  )
}

function UserDiseaseDetail(props: Props & { disease: UserDisease }) {
  const { disease, features, onSelectPair } = props
  const profile = features.profileFor(disease.id)
  const termName = (family: FeatureFamily, term: string) => features.catalogue?.termNames[family].get(term)
  const node: Disease = {
    id: disease.id,
    name: disease.name,
    aliases: [],
    orphaId: disease.orpha_id ?? null,
    candidateOrphaIds: [],
    status: disease.status,
    degree: 0,
    origin: 'user',
  }

  return (
    <div className="panel-body">
      <section className="panel-section">
        <p className="eyebrow">
          <span className="user-mark" aria-hidden>
            ◆
          </span>{' '}
          Added by you
        </p>
        <h2>{disease.name}</h2>
        <p className="meta-row">
          <OrphaLinks disease={node} />
          {disease.status !== 'unknown' && <span className="status">{capitalise(disease.status)}</span>}
          <button type="button" className="link-button" onClick={() => features.onEditUser(disease)}>
            Edit
          </button>
          <button type="button" className="link-button" onClick={() => features.onRemoveUser(disease)}>
            Remove
          </button>
        </p>
        {disease.notes && <p className="muted small">{disease.notes}</p>}
      </section>

      <section className="panel-section">
        <h3>Most similar diseases</h3>
        {features.catalogue === null && (
          <p className="callout small">
            Only your other added diseases are compared until catalogue feature profiles are generated.
          </p>
        )}
        <FeatureNeighbours id={disease.id} disease={node} features={features} limit={20} onSelectPair={onSelectPair} />
      </section>

      <section className="panel-section">
        <h3>Features entered</h3>
        <dl className="feature-list">
          {SET_FAMILIES.map((family) => {
            const values = profile?.sets[family] ? [...profile.sets[family]!] : []
            const raw = disease[FAMILY_COLUMNS[family] as FamilyColumn] as string[]
            return (
              <div key={family}>
                <dt>{FAMILY_TITLES[family]}</dt>
                <dd>
                  {values.length === 0 ? (
                    <span className="no-evidence">{raw.length ? 'None recognised' : 'Not given'}</span>
                  ) : (
                    <span className="shared-terms">
                      {values.map((v) => (
                        <span key={v} className="chip" title={termName(family, v) ?? v}>
                          {v}
                          {termName(family, v) && <span className="term-name"> {termName(family, v)}</span>}
                        </span>
                      ))}
                    </span>
                  )}
                </dd>
              </div>
            )
          })}
          <div>
            <dt>Prevalence</dt>
            <dd>
              {disease.prevalence_estimated_per_person ? (
                `1 in ${Math.round(1 / disease.prevalence_estimated_per_person).toLocaleString()}`
              ) : disease.prevalence_class ? (
                disease.prevalence_class
              ) : (
                <span className="no-evidence">Not given</span>
              )}
            </dd>
          </div>
        </dl>
      </section>
    </div>
  )
}

function PairDetail(props: Props & { a: string; b: string }) {
  const { a, b, graph, diseases, edgesById, filters, features, onSelectDisease } = props
  const edge = edgesById.get(pairId(a, b))
  const view = edge ? (viewEdge(edge, filters) ?? viewEdge(edge, { ...filters, measure: 'overall' })) : null
  const nodeFor = (id: string): Disease | undefined => {
    const known = diseases.get(id)
    if (known) return known
    const user = features.userDiseases.get(id)
    const profile = features.profileFor(id)
    if (!user && !profile) return undefined
    return {
      id,
      name: user?.name ?? profile!.name,
      aliases: [],
      orphaId: user ? (user.orpha_id ?? null) : id,
      candidateOrphaIds: [],
      status: user?.status ?? 'unknown',
      degree: 0,
      origin: user ? 'user' : 'catalogue',
    }
  }
  const da = nodeFor(edge?.source ?? a)
  const db = nodeFor(edge?.target ?? b)
  const pa = features.profileFor(da?.id ?? a)
  const pb = features.profileFor(db?.id ?? b)
  const computed = useMemo(() => (pa && pb ? computeSimilarity(pa, pb, features.weights) : null), [pa, pb, features.weights])
  const unavailable = !pa ? missingProfileReason(da, features) : !pb ? missingProfileReason(db, features) : undefined

  const nameButton = (d: Disease | undefined, fallback: string) => (
    <button type="button" className="link-button" onClick={() => onSelectDisease(d?.id ?? fallback)}>
      {d?.origin === 'user' && (
        <span className="user-mark" aria-hidden>
          ◆{' '}
        </span>
      )}
      {displayName(d?.name ?? fallback)}
    </button>
  )

  return (
    <div className="panel-body">
      <section className="panel-section">
        <p className="eyebrow">Disease pair</p>
        <h2 className="pair-heading">
          {nameButton(da, a)}
          <span className="pair-sep" aria-hidden>
            ↔
          </span>
          {nameButton(db, b)}
        </h2>
        <p className="meta-row small">
          {da && <OrphaLinks disease={da} />}
          {db && <OrphaLinks disease={db} />}
        </p>
      </section>

      {!view ? (
        <section className="panel-section">
          <p className="muted">No literature evidence for this pair from the selected sources.</p>
        </section>
      ) : (
        <>
          <section className="panel-section">
            <div className="stats">
              <div className="stat">
                <span className="stat-value">{score(view.agg.similarity)}</span>
                <span className="stat-label">Overall similarity</span>
              </div>
              <div className="stat">
                <span className="stat-value">
                  {view.agg.evidenceScore}
                  <small>/15</small>
                </span>
                <span className="stat-label">Evidence strength</span>
              </div>
              <div className="stat">
                <span className="stat-value">{view.agg.nPapers}</span>
                <span className="stat-label">{view.agg.nPapers === 1 ? 'Paper' : 'Papers'}</span>
              </div>
            </div>
            <p className="meta-row">
              <span className={`line-key rel-${view.agg.relationship.replace(/ /g, '-')}`} aria-hidden />
              <span>
                Mostly <strong>{view.agg.relationship}</strong>
              </span>
            </p>
            <p className="muted small">Best study: {view.agg.bestDesign}</p>
            {view.agg.flags.length > 0 && (
              <ul className="caveats" aria-label="Evidence caveats">
                {view.agg.flags.map((f) => (
                  <li key={f}>
                    <span className="caveat-icon" aria-hidden>
                      !
                    </span>
                    {capitalise(f)}
                  </li>
                ))}
              </ul>
            )}
          </section>

          <section className="panel-section">
            <h3>Similarity by dimension</h3>
            <DimensionBars
              dimensions={view.agg.dimensions}
              highlight={filters.measure === 'overall' ? undefined : filters.measure}
            />
          </section>
        </>
      )}

      <ComputedSimilarityCard result={computed} unavailable={unavailable} catalogue={features.catalogue} />

      {view && (
        <section className="panel-section">
          <h3>Papers</h3>
          <PaperList papers={view.papers} graph={graph} />
        </section>
      )}
    </div>
  )
}
