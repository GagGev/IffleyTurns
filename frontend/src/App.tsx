import { useCallback, useEffect, useMemo, useState } from 'react'
import { AddDiseaseDialog, type DialogMode, type SavedDisease } from './components/AddDiseaseDialog'
import { DetailPanel, type UserContext } from './components/DetailPanel'
import { EdgeTable } from './components/EdgeTable'
import { FilterPanel } from './components/FilterPanel'
import { GraphView } from './components/GraphView'
import { Legend } from './components/Legend'
import { SearchBox } from './components/SearchBox'
import { UserDiseasesPanel } from './components/UserDiseasesPanel'
import { apiHealth, detailFromPlacement, GraphNotBuiltError, loadGraph, placeDisease } from './data/source'
import type { ApiHealth, EdgeDetail, GraphData, GraphEdge, GraphNode } from './data/types'
import { exportCsv } from './lib/export'
import { download, plural } from './lib/format'
import { DEFAULT_FILTERS, type Filters, filterEdges, findEdge, neighbourhood } from './lib/graph'
import { useHashSelection } from './lib/selection'
import { useCanvasColours } from './lib/theme'
import { exportJson, type UserDisease, useUserDiseases } from './lib/userDiseases'

type View = 'graph' | 'table'
type Scope = 'all' | 'neighbourhood'

export default function App() {
  const [graph, setGraph] = useState<GraphData | null>(null)
  const [loadError, setLoadError] = useState<'not-built' | string | null>(null)

  useEffect(() => {
    loadGraph().then(setGraph, (e: Error) => setLoadError(e instanceof GraphNotBuiltError ? 'not-built' : e.message))
  }, [])

  if (loadError === 'not-built') return <SetupNeeded />
  if (loadError) {
    return (
      <div className="app-message">
        <h1>Could not load the graph</h1>
        <p>{loadError}</p>
      </div>
    )
  }
  if (!graph) {
    return (
      <div className="app-message" aria-live="polite">
        Loading the similarity graph…
      </div>
    )
  }
  return <Explorer graph={graph} />
}

function SetupNeeded() {
  return (
    <div className="app-message">
      <h1>The v2 graph has not been built yet</h1>
      <p>The explorer shows the graph produced by the v2 model. From the repository root:</p>
      <pre className="setup">
        {`python download_databases.py
python generate_features.py
python v2/run_evaluation.py
python v2/build_graph.py
python3 frontend/scripts/build_graph_data.py`}
      </pre>
      <p>
        Then reload this page. To place new diseases, also run <code>python frontend/api/server.py</code>.
      </p>
    </div>
  )
}

/** Poll the placement service until it is ready, then stop. */
function useApiHealth(): ApiHealth {
  const [health, setHealth] = useState<ApiHealth>({ status: 'offline' })
  useEffect(() => {
    let cancelled = false
    let timer: ReturnType<typeof setTimeout>
    const check = async () => {
      const h = await apiHealth()
      if (cancelled) return
      setHealth(h)
      if (h.status !== 'ready') timer = setTimeout(check, h.status === 'loading' ? 2000 : 8000)
    }
    void check()
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [])
  return health
}

function Explorer({ graph }: { graph: GraphData }) {
  const [filters, setFilters] = useState<Filters>(DEFAULT_FILTERS)
  const [selection, select] = useHashSelection()
  const [view, setView] = useState<View>('graph')
  const [scope, setScope] = useState<Scope>('all')
  const [depth, setDepth] = useState(1)
  const colours = useCanvasColours()
  const api = useApiHealth()

  // --- Added diseases and their placements ------------------------------------------
  const user = useUserDiseases()
  const [placing, setPlacing] = useState<Set<string>>(new Set())
  const [placeErrors, setPlaceErrors] = useState<Map<string, string>>(new Map())
  const [linksPerDisease, setLinksPerDisease] = useState(10)
  const [dialog, setDialog] = useState<DialogMode>(null)

  const place = useCallback(
    async (disease: UserDisease, all: UserDisease[]) => {
      setPlacing((s) => new Set(s).add(disease.id))
      setPlaceErrors((m) => {
        const next = new Map(m)
        next.delete(disease.id)
        return next
      })
      try {
        const others = all.filter((d) => d.id !== disease.id).map((d) => ({ id: d.id, disease: d.input }))
        const placement = await placeDisease(disease.id, disease.input, others)
        user.update(disease.id, { placement })
      } catch (e) {
        setPlaceErrors((m) => new Map(m).set(disease.id, (e as Error).message))
      } finally {
        setPlacing((s) => {
          const next = new Set(s)
          next.delete(disease.id)
          return next
        })
      }
    },
    [user],
  )

  const addDiseases = async (items: SavedDisease[]) => {
    const created = user.add(items)
    if (created.length === 1) select({ kind: 'disease', id: created[0].id })
    if (api.status !== 'ready') return
    const all = [...user.diseases, ...created]
    for (const d of created) await place(d, all)
  }

  const updateDisease = async (id: string, item: SavedDisease) => {
    const existing = user.diseases.find((d) => d.id === id)
    if (!existing) return
    const changed: UserDisease = { ...existing, input: item.input, labels: item.labels, placement: undefined }
    user.update(id, { input: item.input, labels: item.labels, placement: undefined })
    if (api.status === 'ready') await place(changed, user.diseases.map((d) => (d.id === id ? changed : d)))
  }

  const placeAll = async () => {
    for (const d of user.diseases) await place(d, user.diseases)
  }

  const removeDisease = (d: UserDisease) => {
    if (!window.confirm(`Remove “${d.input.name}” from your diseases?`)) return
    user.remove(d.id)
    if (selection && (selection.kind === 'disease' ? selection.id === d.id : selection.a === d.id || selection.b === d.id)) {
      select(null)
    }
  }

  // --- Graph plus added diseases ----------------------------------------------------
  const graphNodes = useMemo(() => new Map(graph.nodes.map((n) => [n.id, n])), [graph])
  const { userNodes, userEdges, userDetails } = useMemo(() => {
    const userIds = new Set(user.diseases.map((d) => d.id))
    const nodes: GraphNode[] = []
    const edges = new Map<string, GraphEdge>()
    const details = new Map<string, EdgeDetail>()
    for (const d of user.diseases) {
      for (const n of d.placement?.neighbours.slice(0, linksPerDisease) ?? []) {
        if (!graphNodes.has(n.id) && !userIds.has(n.id)) continue
        const key = [d.id, n.id].sort().join('|')
        const existing = edges.get(key)
        if (existing && existing.score >= n.score) continue
        const id = `${d.id}|${n.id}`
        const main = n.explanation[0]?.modality ?? null
        edges.set(key, {
          id,
          source: d.id,
          target: n.id,
          score: n.score,
          percentile: n.percentile,
          support: n.support,
          mutual: false,
          mainModality: main,
          origin: 'user',
        })
        details.set(id, detailFromPlacement(n, graph.modalities))
      }
    }
    const degree = new Map<string, number>()
    for (const e of edges.values()) {
      degree.set(e.source, (degree.get(e.source) ?? 0) + 1)
      degree.set(e.target, (degree.get(e.target) ?? 0) + 1)
    }
    for (const d of user.diseases) {
      nodes.push({
        id: d.id,
        name: d.input.name,
        category: 'Added by you',
        disorderType: '',
        modalities: d.placement?.present ?? [],
        degree: degree.get(d.id) ?? 0,
        origin: 'user',
      })
    }
    return { userNodes: nodes, userEdges: [...edges.values()], userDetails: details }
  }, [user.diseases, linksPerDisease, graphNodes, graph.modalities])

  const nodes = useMemo(() => {
    const map = new Map(graphNodes)
    for (const n of userNodes) map.set(n.id, n)
    return map
  }, [graphNodes, userNodes])
  const allEdges = useMemo(() => [...graph.edges, ...userEdges], [graph, userEdges])
  const edgesById = useMemo(() => new Map(allEdges.map((e) => [e.id, e])), [allEdges])
  const graphEdgesByNode = useMemo(() => {
    const map = new Map<string, GraphEdge[]>()
    for (const e of graph.edges) {
      ;(map.get(e.source) ?? map.set(e.source, []).get(e.source)!).push(e)
      ;(map.get(e.target) ?? map.set(e.target, []).get(e.target)!).push(e)
    }
    return map
  }, [graph])
  const edgesByNode = useMemo(() => {
    const map = new Map(graphEdgesByNode)
    for (const e of userEdges) {
      for (const id of [e.source, e.target]) map.set(id, [...(map.get(id) ?? []), e])
    }
    return map
  }, [graphEdgesByNode, userEdges])
  const searchable = useMemo(() => [...graph.nodes, ...userNodes], [graph, userNodes])

  // --- Filtering, scope and highlighting --------------------------------------------
  const visible = useMemo(() => filterEdges(allEdges, filters, nodes), [allEdges, filters, nodes])
  const visibleIds = useMemo(() => new Set(visible.map((e) => e.id)), [visible])
  const centres = useMemo(
    () => (!selection ? [] : selection.kind === 'disease' ? [selection.id] : [selection.a, selection.b]),
    [selection],
  )
  const focused = scope === 'neighbourhood' && centres.length > 0
  const scoped = useMemo(() => {
    if (!focused) return visible
    const keep = neighbourhood(visible, centres, depth)
    return visible.filter((e) => keep.has(e.source) && keep.has(e.target))
  }, [focused, visible, centres, depth])

  const selectedEdgeId = selection?.kind === 'pair' ? (findEdge(edgesById, selection.a, selection.b)?.id ?? null) : null
  const highlight = useMemo(() => {
    if (!selection) return { nodes: null, edges: null }
    if (selection.kind === 'pair') return { nodes: new Set(centres), edges: new Set(selectedEdgeId ? [selectedEdgeId] : []) }
    const n = new Set(centres)
    const e = new Set<string>()
    for (const edge of scoped) {
      if (edge.source === selection.id || edge.target === selection.id) {
        e.add(edge.id)
        n.add(edge.source)
        n.add(edge.target)
      }
    }
    return { nodes: n, edges: e }
  }, [selection, centres, scoped, selectedEdgeId])

  const diseaseCount = useMemo(() => {
    const ids = new Set<string>()
    for (const e of scoped) {
      ids.add(e.source)
      ids.add(e.target)
    }
    return ids.size
  }, [scoped])

  const fitKey = focused ? `n-${depth}-${centres.join(',')}` : 'all'
  const selectDisease = (id: string) => select({ kind: 'disease', id })
  const selectPair = (a: string, b: string) => select({ kind: 'pair', a, b })

  const userContext: UserContext = {
    diseases: new Map(user.diseases.map((d) => [d.id, d])),
    api,
    placing,
    errors: placeErrors,
    edgeDetails: userDetails,
    onEdit: (disease) => setDialog({ kind: 'edit', disease }),
    onRemove: removeDisease,
    onPlace: (id) => {
      const d = user.diseases.find((x) => x.id === id)
      if (d) void place(d, user.diseases)
    },
  }

  const apiLabel = { ready: 'ready', loading: 'loading model', error: 'error', offline: 'not running' }[api.status]

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>Rare Disease Similarity Explorer</h1>
          <p className="muted small">
            v2 model · {plural(graph.stats.nodes, 'disease')} · {plural(graph.stats.edges, 'edge')}
          </p>
        </div>
        <span
          className={`backend-status ${api.status === 'ready' ? 'is-on' : ''}`}
          title={api.error ?? 'Local service that places new diseases with the v2 model'}
        >
          <span className="dot" aria-hidden />
          Placement service: {apiLabel}
        </span>
      </header>

      <aside className="sidebar" aria-label="Search and filters">
        <SearchBox diseases={searchable} onSelect={selectDisease} />
        <UserDiseasesPanel
          diseases={user.diseases}
          api={api}
          placing={placing}
          linksPerDisease={linksPerDisease}
          selectedId={selection?.kind === 'disease' ? selection.id : null}
          onLinksChange={setLinksPerDisease}
          onAdd={() => setDialog({ kind: 'add' })}
          onSelect={selectDisease}
          onEdit={(disease) => setDialog({ kind: 'edit', disease })}
          onRemove={removeDisease}
          onPlaceAll={() => void placeAll()}
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
            {plural(scoped.length, 'edge')} · {plural(diseaseCount, 'disease')}
          </span>
          <button type="button" className="button" disabled={scoped.length === 0} onClick={() => exportCsv(scoped, nodes)}>
            Export CSV
          </button>
        </div>

        <div className="main-content">
          {view === 'graph' ? (
            <GraphView
              nodes={nodes}
              edges={scoped}
              pinnedIds={centres}
              selection={selection}
              highlightNodes={highlight.nodes}
              highlightEdges={highlight.edges}
              fixedLayout={graph.hasLayout}
              fitKey={fitKey}
              colours={colours}
              onSelectDisease={selectDisease}
              onSelectPair={selectPair}
              onClear={() => select(null)}
            />
          ) : (
            <EdgeTable edges={scoped} nodes={nodes} selectedEdgeId={selectedEdgeId} onSelectPair={selectPair} />
          )}
          {scoped.length === 0 && view === 'graph' && <p className="empty-overlay">No edges match the current filters.</p>}
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
          nodes={nodes}
          edgesById={edgesById}
          edgesByNode={edgesByNode}
          visible={visible}
          visibleIds={visibleIds}
          selection={selection}
          user={userContext}
          onSelectDisease={selectDisease}
          onSelectPair={selectPair}
        />
      </aside>

      <AddDiseaseDialog
        mode={dialog}
        api={api}
        others={user.diseases}
        onAdd={(items) => void addDiseases(items)}
        onUpdate={(id, item) => void updateDisease(id, item)}
        onClose={() => setDialog(null)}
      />
    </div>
  )
}
