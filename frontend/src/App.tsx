import { useCallback, useEffect, useMemo, useState } from 'react'
import { AddDiseaseDialog, type DialogMode, type SavedDisease } from './components/AddDiseaseDialog'
import { DetailPanel, type UserContext } from './components/DetailPanel'
import { EdgeTable } from './components/EdgeTable'
import { FilterPanel } from './components/FilterPanel'
import { GraphView } from './components/GraphView'
import { Legend } from './components/Legend'
import { SearchBox } from './components/SearchBox'
import { PaperPanel } from './components/PaperPanel'
import { PapersPanel } from './components/PapersPanel'
import { UserDiseasesPanel } from './components/UserDiseasesPanel'
import { type Literature, loadLiterature } from './data/annotations'
import { apiHealth, detailFromPlacement, GraphNotBuiltError, loadGraph, placeDisease } from './data/source'
import type { ApiHealth, DiseaseInput, EdgeDetail, GraphData, GraphEdge, GraphNode, PaperEntry, PaperResult } from './data/types'
import { useNodeColouring } from './lib/colouring'
import { exportCsv } from './lib/export'
import { evaluateLiterature, usePaperEvaluation } from './lib/usePaper'
import {
  inputFromResult,
  mockExtractPaper,
  paperEntryFor,
  parsePaperResult,
  placementFromResult,
} from './lib/paperService'
import { download, plural } from './lib/format'
import { DEFAULT_FILTERS, type Filters, filterEdges, findEdge, neighbourhood } from './lib/graph'
import { useHashSelection } from './lib/selection'
import { categoricalColour, useCanvasColours, verdictColour } from './lib/theme'
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

  // --- Node colour: cluster, category, gene, symptom, onset, inheritance or similarity ---
  const [clusterFocus, setClusterFocus] = useState<number | null>(null)
  const clusterColour = useCallback((index: number) => categoricalColour(index, colours.dark), [colours.dark])
  const categoryIndex = useMemo(() => new Map(graph.categories.map((c, i) => [c, i])), [graph.categories])
  const categoryColour = useCallback(
    (name: string) => categoricalColour(categoryIndex.get(name) ?? 0, colours.dark),
    [categoryIndex, colours.dark],
  )
  const categoryCounts = useMemo(() => {
    const counts = new Map<string, number>()
    for (const n of graph.nodes) if (n.category) counts.set(n.category, (counts.get(n.category) ?? 0) + 1)
    return [...counts].map(([name, count]) => ({ name, count })).sort((a, b) => b.count - a.count)
  }, [graph.nodes])

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
    if (d.paper && d.paper.id === activePaperId) setActivePaperId(null)
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
    // A paper's claims always get their edge, even beyond the links-per-disease limit.
    const shown = (d: UserDisease) => {
      const claimed = new Set(d.paper?.claims.map((c) => c.b))
      return (d.placement?.neighbours ?? []).filter((n, i) => i < linksPerDisease || claimed.has(n.id))
    }
    for (const d of user.diseases) {
      for (const n of shown(d)) {
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
    // An added disease takes the cluster most of its nearest neighbours belong to (weighted by score).
    const inferCluster = (d: UserDisease): number => {
      const votes = new Map<number, number>()
      for (const n of shown(d)) {
        const cluster = graphNodes.get(n.id)?.cluster ?? -1
        if (cluster >= 0) votes.set(cluster, (votes.get(cluster) ?? 0) + Math.max(n.score, 0) + 0.05)
      }
      return [...votes].sort((a, b) => b[1] - a[1])[0]?.[0] ?? -1
    }
    for (const d of user.diseases) {
      nodes.push({
        id: d.id,
        name: d.input.name,
        category: 'Added by you',
        disorderType: '',
        cluster: inferCluster(d),
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
  const userInputs = useMemo(() => new Map<string, DiseaseInput>(user.diseases.map((d) => [d.id, d.input])), [user.diseases])
  const colouring = useNodeColouring({
    nodes,
    userInputs,
    edgesByNode,
    selectedDisease: selection?.kind === 'disease' ? selection.id : null,
    colours,
    defaultMode: graph.clusters.length > 0 ? 'cluster' : 'plain',
    clusterColour,
    categoryColour,
  })
  const colourBy = colouring.mode

  // --- Papers: uploaded ones live with their focal disease; the literature set is a static file ------
  const [literature, setLiterature] = useState<Literature | null>(null)
  const [literatureError, setLiteratureError] = useState<string | null>(null)
  useEffect(() => {
    loadLiterature().then(setLiterature, (e: Error) => setLiteratureError(e.message))
  }, [])
  const [activePaperId, setActivePaperId] = useState<string | null>(null)
  const [uploadStage, setUploadStage] = useState<string | null>(null)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const uploads = useMemo(() => user.diseases.filter((d) => d.paper), [user.diseases])
  const activePaper = useMemo<PaperEntry | null>(
    () => uploads.find((d) => d.paper!.id === activePaperId)?.paper ?? literature?.papers.find((p) => p.id === activePaperId) ?? null,
    [uploads, literature, activePaperId],
  )
  const activeFocal = uploads.find((d) => d.paper!.id === activePaperId)
  const graphEdgesById = useMemo(() => new Map(graph.edges.map((e) => [e.id, e])), [graph])
  const literatureEval = useMemo(
    () => (literature ? evaluateLiterature(literature.papers, graph, graphEdgesById) : null),
    [literature, graph, graphEdgesById],
  )
  const evaluations = usePaperEvaluation(activePaper, graph, edgesById, userDetails)

  const uploadPaper = async (file: File): Promise<boolean> => {
    setUploadError(null)
    setUploadStage('Reading the paper')
    try {
      let result: PaperResult | null = null
      if (/\.json$/i.test(file.name)) {
        const parsed = parsePaperResult(await file.text())
        if (typeof parsed !== 'string') result = parsed
      }
      result ??= await mockExtractPaper(file, { graph, edgesByNode: graphEdgesByNode }, setUploadStage)
      const { input, labels } = inputFromResult(result)
      const created = user.addPaper(input, labels, (id) => ({
        placement: placementFromResult(result, id),
        paper: paperEntryFor(result, id, file.name),
      }))
      setActivePaperId(created.paper!.id)
      select(null)
      return true
    } catch (e) {
      setUploadError(`Could not process ${file.name}: ${(e as Error).message}`)
      return false
    } finally {
      setUploadStage(null)
    }
  }

  // The active paper's claims drawn over the graph: real edges in their verdict colour, and a dashed line
  // between two diseases when the paper links them but the graph does not.
  const overlay = useMemo(() => {
    if (!activePaper) return null
    const tint = new Map<string, string>()
    const extra: GraphEdge[] = []
    const members = new Set<string>()
    // Claims the graph has no edge for can lie far away; keep them out of the camera framing.
    const framed = new Set<string>()
    for (const ev of evaluations) {
      const { a, b } = ev.claim
      if (!nodes.has(a) || !nodes.has(b)) continue
      members.add(a)
      members.add(b)
      const colour = verdictColour(ev.verdict ?? 'edge-only', colours.dark)
      if (ev.edge) {
        framed.add(a)
        framed.add(b)
        tint.set(ev.edge.id, colour)
        extra.push(ev.edge)
      } else {
        const ghost: GraphEdge = {
          id: `claim:${a}|${b}`,
          source: a,
          target: b,
          score: 0,
          percentile: 0,
          support: 'novel',
          mutual: false,
          mainModality: null,
          origin: 'user',
        }
        tint.set(ghost.id, colour)
        extra.push(ghost)
      }
    }
    // A paper that draws no comparisons still has a focal disease: show it with the edges it was placed by.
    const edgeIds = new Set(tint.keys())
    if (evaluations.length === 0 && activeFocal) {
      members.add(activeFocal.id)
      for (const e of userEdges) {
        if (e.source !== activeFocal.id && e.target !== activeFocal.id) continue
        members.add(e.source)
        members.add(e.target)
        edgeIds.add(e.id)
      }
    }
    return { tint, extra, members, edgeIds, ids: [...(framed.size > 0 ? framed : members)] }
  }, [activePaper, evaluations, nodes, colours.dark, activeFocal, userEdges])

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

  const drawn = useMemo(() => {
    if (!overlay) return scoped
    const have = new Set(scoped.map((e) => e.id))
    return [...scoped, ...overlay.extra.filter((e) => !have.has(e.id))]
  }, [scoped, overlay])

  const selectedEdgeId = selection?.kind === 'pair' ? (findEdge(edgesById, selection.a, selection.b)?.id ?? null) : null
  const highlight = useMemo(() => {
    if (!selection && overlay) return { nodes: overlay.members, edges: overlay.edgeIds }
    if (!selection && colourBy !== 'cluster' && colouring.members) {
      // Gene, symptom, onset, inheritance or similarity colouring: keep the diseases that carry it at full strength.
      const inside = new Set(scoped.filter((e) => colouring.members!.has(e.source) && colouring.members!.has(e.target)).map((e) => e.id))
      return { nodes: colouring.members, edges: inside }
    }
    if (!selection && colourBy === 'cluster' && clusterFocus !== null) {
      // No selection: keep one cluster at full strength and dim the rest.
      const members = new Set<string>()
      for (const [id, n] of nodes) if (n.cluster === clusterFocus) members.add(id)
      const inside = new Set(scoped.filter((e) => members.has(e.source) && members.has(e.target)).map((e) => e.id))
      return { nodes: members, edges: inside }
    }
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
  }, [selection, centres, scoped, selectedEdgeId, clusterFocus, nodes, colourBy, colouring.members, overlay])

  const diseaseCount = useMemo(() => {
    const ids = new Set<string>()
    for (const e of drawn) {
      ids.add(e.source)
      ids.add(e.target)
    }
    return ids.size
  }, [drawn])

  const fitKey = focused ? `n-${depth}-${centres.join(',')}` : 'all'
  const onColourBy = (mode: 'gene' | 'symptom' | 'onset' | 'inheritance', term: string | null) => {
    colouring.setMode(mode)
    if (mode === 'gene' || mode === 'symptom') colouring.setTerm(term)
    else colouring.setGroupFocus(term)
    setActivePaperId(null)
  }
  const focusCluster = (cluster: number | null) => {
    if (cluster !== null) colouring.setMode('cluster')
    setClusterFocus(cluster)
  }
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

  // The details sidebar only opens for a selected disease, pair or paper.
  const showDetails = Boolean(selection || activePaper)

  return (
    <div className={`app${showDetails ? ' has-details' : ''}`}>
      <header className="app-header">
        <div>
          <h1>Rare Disease Similarity Explorer</h1>
          <p className="muted small">
            v2 model · {plural(graph.stats.nodes, 'disease')} · {plural(graph.stats.edges, 'edge')} ·{' '}
            {plural(graph.stats.support.novel ?? 0, 'novel hypothesis', 'novel hypotheses')}
          </p>
        </div>
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
        <PapersPanel
          uploads={uploads}
          literature={literature?.papers ?? null}
          literatureError={literatureError}
          literatureEval={literatureEval}
          activeId={activePaperId}
          busy={uploadStage}
          error={uploadError}
          dark={colours.dark}
          onUpload={(file) => void uploadPaper(file)}
          onActivate={(id) => {
            setActivePaperId(id)
            if (id) select(null)
          }}
          onRemoveUpload={removeDisease}
        />
        <FilterPanel graph={graph} filters={filters} onChange={setFilters} />
        <Legend
          colouring={colouring}
          nodes={nodes}
          clusters={graph.clusters}
          modularity={graph.clustering?.modularity ?? null}
          categories={categoryCounts}
          clusterColour={clusterColour}
          categoryColour={categoryColour}
          focusCluster={clusterFocus}
          onFocusCluster={focusCluster}
        />
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
            <button type="button" aria-pressed={!focused} onClick={() => setScope('all')}>
              Whole network
            </button>
            <button
              type="button"
              aria-pressed={focused}
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
          {activePaper && (
            <span className="paper-chip" title={activePaper.title}>
              Paper: {activePaper.title}
              <button type="button" onClick={() => setActivePaperId(null)} aria-label="Hide paper edges">
                ×
              </button>
            </span>
          )}
          <span className="toolbar-count muted small" aria-live="polite">
            {plural(drawn.length, 'edge')} · {plural(diseaseCount, 'disease')}
          </span>
          <button type="button" className="button" disabled={scoped.length === 0} onClick={() => exportCsv(scoped, nodes)}>
            Export CSV
          </button>
        </div>

        <div className="main-content">
          {view === 'graph' ? (
            <GraphView
              nodes={nodes}
              edges={drawn}
              pinnedIds={centres}
              selection={selection}
              highlightNodes={highlight.nodes}
              highlightEdges={highlight.edges}
              fixedLayout={graph.hasLayout && !focused}
              fitAll={focused}
              fitKey={fitKey}
              colours={colours}
              colourOf={colouring.colourOf}
              clusters={graph.clusters}
              edgeAlpha={colourBy === 'plain' ? 0.4 : 0.1}
              edgeTint={overlay?.tint ?? null}
              frameIds={overlay?.ids ?? (colouring.members && colouring.members.size <= 60 && colourBy !== 'cluster' ? [...colouring.members] : null)}
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

      {showDetails && (
      <aside className="details" aria-label="Details">
        {selection && (
          <button type="button" className="close-button" onClick={() => select(null)} aria-label="Clear selection">
            ×
          </button>
        )}
        {activePaper && selection && (
          <button type="button" className="back-to-paper link-button small" onClick={() => select(null)}>
            ← Back to paper: {activePaper.title}
          </button>
        )}
        {activePaper && !selection ? (
          <PaperPanel
            paper={activePaper}
            evaluations={evaluations}
            disease={activeFocal}
            nodes={nodes}
            dark={colours.dark}
            onSelectPair={selectPair}
            onSelectDisease={selectDisease}
            onColourBy={onColourBy}
            onClose={() => setActivePaperId(null)}
          />
        ) : (
        <DetailPanel
          graph={graph}
          nodes={nodes}
          edgesById={edgesById}
          edgesByNode={edgesByNode}
          visible={visible}
          visibleIds={visibleIds}
          selection={selection}
          user={userContext}
          clusterColour={clusterColour}
          focusCluster={clusterFocus}
          onFocusCluster={focusCluster}
          onSelectDisease={selectDisease}
          onSelectPair={selectPair}
        />
        )}
      </aside>
      )}

      <AddDiseaseDialog
        mode={dialog}
        api={api}
        others={user.diseases}
        onAdd={(items) => void addDiseases(items)}
        onUpdate={(id, item) => void updateDisease(id, item)}
        onUploadPaper={uploadPaper}
        paperBusy={uploadStage}
        paperError={uploadError}
        onClose={() => setDialog(null)}
      />
    </div>
  )
}
