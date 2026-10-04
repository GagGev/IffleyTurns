import { useCallback, useEffect, useMemo, useState } from 'react'
import { AddDiseaseDialog, type DialogMode } from './components/AddDiseaseDialog'
import { DetailPanel, type FeatureContext } from './components/DetailPanel'
import { EdgeTable } from './components/EdgeTable'
import { FilterPanel } from './components/FilterPanel'
import { GraphView } from './components/GraphView'
import { Legend } from './components/Legend'
import { SearchBox } from './components/SearchBox'
import { UserDiseasesPanel } from './components/UserDiseasesPanel'
import { DEFAULT_WEIGHTS, type FeatureCatalogue } from './data/features'
import { dataSource } from './data/source'
import type { ComputedEdge, Disease, Edge, LiteratureGraph } from './data/types'
import { exportCsv } from './lib/export'
import { measureLabel, plural } from './lib/format'
import { DEFAULT_FILTERS, type Filters, filterEdges, neighbourhood, pairId } from './lib/graph'
import { useHashSelection } from './lib/selection'
import { rankNeighbours } from './lib/similarity'
import { useCanvasColours } from './lib/theme'
import { exportJson, toProfile, type UserDisease, useUserDiseases } from './lib/userDiseases'

type View = 'graph' | 'table'
type Scope = 'all' | 'neighbourhood'

export default function App() {
  const [graph, setGraph] = useState<LiteratureGraph | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)

  useEffect(() => {
    dataSource.loadLiteratureGraph().then(setGraph, (e: Error) => setLoadError(e.message))
  }, [])

  if (loadError) {
    return (
      <div className="app-message">
        <h1>Could not load data</h1>
        <p>{loadError}</p>
      </div>
    )
  }
  if (!graph) {
    return (
      <div className="app-message" aria-live="polite">
        Loading literature graph…
      </div>
    )
  }
  return <Explorer graph={graph} />
}

const userNode = (d: UserDisease): Disease => ({
  id: d.id,
  name: d.name,
  aliases: [],
  orphaId: d.orpha_id ?? null,
  candidateOrphaIds: [],
  status: d.status,
  degree: 0,
  origin: 'user',
})

const catalogueNode = (id: string, name: string): Disease => ({
  id,
  name,
  aliases: [],
  orphaId: id,
  candidateOrphaIds: [],
  status: 'unknown',
  degree: 0,
  origin: 'catalogue',
})

function download(name: string, text: string, type: string) {
  const url = URL.createObjectURL(new Blob([text], { type }))
  const a = document.createElement('a')
  a.href = url
  a.download = name
  a.click()
  URL.revokeObjectURL(url)
}

function Explorer({ graph }: { graph: LiteratureGraph }) {
  const [filters, setFilters] = useState<Filters>(DEFAULT_FILTERS)
  const [selection, select] = useHashSelection()
  const [view, setView] = useState<View>('graph')
  const [scope, setScope] = useState<Scope>('all')
  const [depth, setDepth] = useState(1)
  const colours = useCanvasColours()

  // --- Feature profiles and the user's own diseases ---------------------------
  const [catalogue, setCatalogue] = useState<FeatureCatalogue | null | undefined>(undefined)
  useEffect(() => {
    dataSource.loadFeatureCatalogue().then(setCatalogue, () => setCatalogue(null))
  }, [])
  const user = useUserDiseases()
  const [linksPerDisease, setLinksPerDisease] = useState(5)
  const [dialog, setDialog] = useState<DialogMode>(null)
  const weights = catalogue?.weights ?? DEFAULT_WEIGHTS

  const userById = useMemo(() => new Map(user.diseases.map((d) => [d.id, d])), [user.diseases])
  const userProfiles = useMemo(() => user.diseases.map((d) => toProfile(d, catalogue ?? null)), [user.diseases, catalogue])
  const allProfiles = useMemo(() => [...(catalogue?.profiles ?? []), ...userProfiles], [catalogue, userProfiles])

  const literatureDiseases = useMemo(() => new Map<string, Disease>(graph.nodes.map((n) => [n.id, n])), [graph])
  const profileFor = useCallback(
    (id: string) => {
      const own = userProfiles.find((p) => p.id === id)
      if (own) return own
      const orpha = literatureDiseases.get(id)?.orphaId ?? id
      return catalogue?.byId.get(orpha)
    },
    [userProfiles, literatureDiseases, catalogue],
  )

  /** Each added disease links to its most similar diseases by features. */
  const computedEdges = useMemo(() => {
    const out = new Map<string, ComputedEdge>()
    for (const profile of userProfiles) {
      for (const n of rankNeighbours(profile, allProfiles, linksPerDisease, weights)) {
        const id = pairId(profile.id, n.profile.id)
        if (!out.has(id)) out.set(id, { id, source: profile.id, target: n.profile.id, similarity: n.similarity })
      }
    }
    return [...out.values()]
  }, [userProfiles, allProfiles, linksPerDisease, weights])

  const centres = useMemo(
    () => (!selection ? [] : selection.kind === 'disease' ? [selection.id] : [selection.a, selection.b]),
    [selection],
  )

  // Literature diseases, plus added diseases and any catalogue disease that is
  // linked or selected.
  const diseases = useMemo(() => {
    const map = new Map(literatureDiseases)
    for (const d of user.diseases) map.set(d.id, userNode(d))
    for (const id of [...computedEdges.flatMap((e) => [e.source, e.target]), ...centres]) {
      const profile = !map.has(id) ? catalogue?.byId.get(id) : undefined
      if (profile) map.set(id, catalogueNode(id, profile.name))
    }
    return map
  }, [literatureDiseases, user.diseases, computedEdges, centres, catalogue])

  const searchable = useMemo(() => {
    const list = [...graph.nodes, ...user.diseases.map(userNode)]
    for (const p of catalogue?.profiles ?? []) if (!literatureDiseases.has(p.id)) list.push(catalogueNode(p.id, p.name))
    return list
  }, [graph, user.diseases, catalogue, literatureDiseases])

  // --- Literature edges --------------------------------------------------------
  const edgesById = useMemo(() => new Map<string, Edge>(graph.edges.map((e) => [e.id, e])), [graph])
  const edgesByDisease = useMemo(() => {
    const map = new Map<string, Edge[]>()
    for (const e of graph.edges) {
      map.set(e.source, [...(map.get(e.source) ?? []), e])
      map.set(e.target, [...(map.get(e.target) ?? []), e])
    }
    return map
  }, [graph])

  const visible = useMemo(() => filterEdges(graph, filters, literatureDiseases), [graph, filters, literatureDiseases])
  const visibleIds = useMemo(() => new Set(visible.map((v) => v.edge.id)), [visible])
  const focused = scope === 'neighbourhood' && centres.length > 0

  const { scoped, scopedComputed } = useMemo(() => {
    if (!focused) return { scoped: visible, scopedComputed: computedEdges }
    const keep = neighbourhood([...visible.map((v) => v.edge), ...computedEdges], centres, depth)
    const inside = (e: { source: string; target: string }) => keep.has(e.source) && keep.has(e.target)
    return { scoped: visible.filter((v) => inside(v.edge)), scopedComputed: computedEdges.filter(inside) }
  }, [focused, visible, computedEdges, centres, depth])

  const highlight = useMemo(() => {
    if (!selection) return { nodes: null, edges: null }
    if (selection.kind === 'pair') {
      return { nodes: new Set(centres), edges: new Set([pairId(selection.a, selection.b)]) }
    }
    const nodes = new Set(centres)
    const edges = new Set<string>()
    for (const e of [...scoped.map((v) => v.edge), ...scopedComputed]) {
      if (e.source === selection.id || e.target === selection.id) {
        edges.add(e.id)
        nodes.add(e.source)
        nodes.add(e.target)
      }
    }
    return { nodes, edges }
  }, [selection, centres, scoped, scopedComputed])

  const diseaseCount = useMemo(() => {
    const ids = new Set<string>()
    for (const e of [...scoped.map((v) => v.edge), ...scopedComputed]) {
      ids.add(e.source)
      ids.add(e.target)
    }
    return ids.size
  }, [scoped, scopedComputed])

  const fitKey = focused ? `n-${depth}-${centres.join(',')}` : 'all'
  const label = measureLabel(filters.measure)
  const selectDisease = (id: string) => select({ kind: 'disease', id })
  const selectPair = (a: string, b: string) => select({ kind: 'pair', a, b })

  const removeUserDisease = (d: UserDisease) => {
    if (!window.confirm(`Remove “${d.name}” from your diseases?`)) return
    user.remove(d.id)
    if (centres.includes(d.id)) select(null)
  }

  const features: FeatureContext = {
    catalogue,
    profiles: allProfiles,
    profileFor,
    weights,
    userDiseases: userById,
    onEditUser: (disease) => setDialog({ kind: 'edit', disease }),
    onRemoveUser: removeUserDisease,
  }

  const profileStatus =
    catalogue === undefined
      ? 'loading'
      : catalogue === null
        ? 'not generated'
        : `${catalogue.profiles.length.toLocaleString()} diseases`

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>Rare Disease Relationship Explorer</h1>
          <p className="muted small">
            {plural(graph.stats.nodes, 'disease')} · {plural(graph.stats.edges, 'literature-backed pair')} ·{' '}
            {plural(graph.stats.paperRows, 'paper row')}
          </p>
        </div>
        <span className={`backend-status ${catalogue ? 'is-on' : ''}`} title="Feature profiles used for computed similarity">
          <span className="dot" aria-hidden />
          Feature profiles: {profileStatus}
        </span>
      </header>

      <aside className="sidebar" aria-label="Search and filters">
        <SearchBox diseases={searchable} onSelect={selectDisease} />
        <UserDiseasesPanel
          diseases={user.diseases}
          catalogue={catalogue}
          linksPerDisease={linksPerDisease}
          selectedId={selection?.kind === 'disease' ? selection.id : null}
          onLinksChange={setLinksPerDisease}
          onAdd={() => setDialog({ kind: 'add' })}
          onSelect={selectDisease}
          onEdit={(disease) => setDialog({ kind: 'edit', disease })}
          onRemove={removeUserDisease}
          onExport={() => download('my-diseases.json', exportJson(user.diseases), 'application/json')}
        />
        <FilterPanel graph={graph} filters={filters} onChange={setFilters} />
        <Legend />
      </aside>

      <main className="main">
        <div className="toolbar">
          <div className="segmented" role="tablist" aria-label="View">
            {(['graph', 'table'] as View[]).map((v) => (
              <button key={v} type="button" role="tab" aria-selected={view === v} onClick={() => setView(v)}>
                {v === 'graph' ? 'Graph' : 'Table'}
              </button>
            ))}
          </div>
          <div className="segmented" aria-label="Scope">
            <button type="button" aria-pressed={scope === 'all'} onClick={() => setScope('all')}>
              Whole network
            </button>
            <button
              type="button"
              aria-pressed={scope === 'neighbourhood'}
              disabled={centres.length === 0}
              title={centres.length === 0 ? 'Select a disease or pair first' : undefined}
              onClick={() => setScope('neighbourhood')}
            >
              Neighbourhood
            </button>
          </div>
          {focused && (
            <label className="inline-field">
              Depth
              <select value={depth} onChange={(e) => setDepth(Number(e.target.value))}>
                <option value={1}>1 step</option>
                <option value={2}>2 steps</option>
                <option value={3}>3 steps</option>
              </select>
            </label>
          )}
          <span className="toolbar-count muted small" aria-live="polite">
            {plural(scoped.length, 'pair')}
            {scopedComputed.length > 0 && ` + ${scopedComputed.length} computed`} · {plural(diseaseCount, 'disease')}
          </span>
          <button
            type="button"
            className="button"
            disabled={scoped.length === 0}
            onClick={() => exportCsv(scoped, diseases, filters.measure)}
          >
            Export CSV
          </button>
        </div>

        <div className="main-content">
          {view === 'graph' ? (
            <GraphView
              diseases={diseases}
              edges={scoped}
              computedEdges={scopedComputed}
              pinnedIds={centres}
              selection={selection}
              highlightNodes={highlight.nodes}
              highlightEdges={highlight.edges}
              measureLabel={label}
              fitKey={fitKey}
              colours={colours}
              onSelectDisease={selectDisease}
              onSelectPair={selectPair}
              onClear={() => select(null)}
            />
          ) : (
            <EdgeTable
              edges={scoped}
              diseases={diseases}
              selection={selection}
              measureLabel={label}
              onSelectPair={selectPair}
            />
          )}
          {scoped.length === 0 && scopedComputed.length === 0 && view === 'graph' && (
            <p className="empty-overlay">No pairs match the current filters.</p>
          )}
        </div>
      </main>

      <aside className="details" aria-label="Details">
        {selection && (
          <button type="button" className="close-button" onClick={() => select(null)} aria-label="Clear selection">
            ×
          </button>
        )}
        <DetailPanel
          graph={graph}
          diseases={diseases}
          edgesById={edgesById}
          edgesByDisease={edgesByDisease}
          visible={visible}
          visibleIds={visibleIds}
          filters={filters}
          selection={selection}
          features={features}
          onSelectDisease={selectDisease}
          onSelectPair={selectPair}
        />
      </aside>

      <AddDiseaseDialog
        mode={dialog}
        catalogue={catalogue ?? null}
        candidates={allProfiles}
        weights={weights}
        onAdd={(items) => {
          const created = user.add(items)
          if (created.length === 1) selectDisease(created[0].id)
        }}
        onUpdate={user.update}
        onClose={() => setDialog(null)}
      />
    </div>
  )
}
