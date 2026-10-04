import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import ForceGraph2D from 'react-force-graph-2d'
import type { ForceGraphMethods, LinkObject, NodeObject } from 'react-force-graph-2d'
import { edgeShard } from '../data/source'
import type { ClusterInfo, GraphEdge, GraphNode } from '../data/types'
import { clusterId, displayName, modalityLabel, percentile, plural } from '../lib/format'
import { isClaimOnly } from '../lib/graph'
import type { Selection } from '../lib/selection'
import { type CanvasColours, withAlpha } from '../lib/theme'

interface GNode {
  id: string
  disease: GraphNode
  degree: number
}
interface GLink {
  edge: GraphEdge
}
type N = NodeObject<GNode>
type L = LinkObject<GNode, GLink>

interface Props {
  nodes: Map<string, GraphNode>
  edges: GraphEdge[]
  /** Diseases to draw even without visible edges (the current selection). */
  pinnedIds: string[]
  selection: Selection
  /** Diseases to keep at full strength; everything else is dimmed. */
  highlightNodes: Set<string> | null
  highlightEdges: Set<string> | null
  /** Use the precomputed layout instead of simulating one. */
  fixedLayout: boolean
  /** Changes whenever the graph should re-fit to the viewport. */
  fitKey: string
  colours: CanvasColours
  /** Fill colour of a disease (by cluster, category or plain). */
  colourOf: (disease: GraphNode) => string
  clusters: ClusterInfo[]
  /** Opacity of edges that are not highlighted; low when nodes carry the colour. */
  edgeAlpha: number
  /** Edges drawn in a given colour regardless of highlighting (the claims of the active paper). */
  edgeTint: Map<string, string> | null
  /** Diseases to bring into view when nothing is selected (the diseases a paper concerns). */
  frameIds: string[] | null
  onSelectDisease: (id: string) => void
  onSelectPair: (a: string, b: string) => void
  onClear: () => void
}

type Hover = { kind: 'node'; node: N } | { kind: 'link'; link: L } | null

/**
 * Layout state owned by the graph engine, which writes x/y into these node
 * objects.  Kept outside React state on purpose: reusing the objects keeps
 * positions stable when filters change.
 */
class NodeStore {
  private nodes = new Map<string, N>()
  private records = new Map<string, GraphNode>()

  sync(records: Map<string, GraphNode>) {
    if (records === this.records) return
    this.records = records
    for (const [id, node] of this.nodes) {
      const d = records.get(id)
      if (d) node.disease = d
    }
  }

  node(id: string, degree: number, fixed: boolean): N | undefined {
    const disease = this.records.get(id)
    if (!disease) return undefined
    let node = this.nodes.get(id)
    if (!node) {
      node = { id, disease, degree }
      if (disease.x !== undefined && disease.y !== undefined) {
        node.x = disease.x
        node.y = disease.y
        if (fixed) {
          node.fx = disease.x
          node.fy = disease.y
        }
      }
      this.nodes.set(id, node)
    }
    node.degree = degree
    return node
  }

  get(id: string): N | undefined {
    return this.nodes.get(id)
  }
}

const nodeRadius = (degree: number) => 2 + Math.sqrt(degree) * 0.6
const endId = (end: L['source']) => (typeof end === 'object' ? (end as N).id : String(end))

function diamond(ctx: CanvasRenderingContext2D, x: number, y: number, d: number) {
  ctx.moveTo(x, y - d)
  ctx.lineTo(x + d, y)
  ctx.lineTo(x, y + d)
  ctx.lineTo(x - d, y)
  ctx.closePath()
}

export function GraphView(props: Props) {
  const { nodes, edges, pinnedIds, selection, highlightNodes, highlightEdges, colours, fixedLayout, colourOf, clusters, edgeAlpha, edgeTint, frameIds } = props
  const wrapper = useRef<HTMLDivElement>(null)
  const graph = useRef<ForceGraphMethods<N, L>>(undefined)
  const [store] = useState(() => new NodeStore())
  const fitPending = useRef(true)
  const [size, setSize] = useState({ width: 0, height: 0 })
  const [hover, setHover] = useState<Hover>(null)
  const [pointer, setPointer] = useState({ x: 0, y: 0 })

  useEffect(() => {
    const el = wrapper.current
    if (!el) return
    const observer = new ResizeObserver(([entry]) => {
      setSize({ width: entry.contentRect.width, height: entry.contentRect.height })
    })
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  // Only pinned diseases that no visible edge already draws change the graph
  // data; otherwise selecting a node would rebuild it.
  const extraPinned = useMemo(() => {
    const drawn = new Set<string>()
    for (const { source, target } of edges) {
      drawn.add(source)
      drawn.add(target)
    }
    return pinnedIds.filter((id) => !drawn.has(id)).join('\n')
  }, [edges, pinnedIds])

  store.sync(nodes)
  const data = useMemo(() => {
    const degree = new Map<string, number>()
    for (const { source, target } of edges) {
      degree.set(source, (degree.get(source) ?? 0) + 1)
      degree.set(target, (degree.get(target) ?? 0) + 1)
    }
    for (const id of extraPinned ? extraPinned.split('\n') : []) {
      const record = nodes.get(id)
      if (!fixedLayout || record?.x !== undefined) degree.set(id, 0)
    }
    const out: N[] = []
    for (const [id, d] of degree) {
      const node = store.node(id, d, fixedLayout)
      if (node) out.push(node)
    }
    // Added diseases have no precomputed position: put them beside their matches.
    if (fixedLayout) {
      for (const node of out) {
        if (node.fx !== undefined) continue
        const linked = edges
          .filter((e) => e.source === node.id || e.target === node.id)
          .map((e) => store.get(e.source === node.id ? e.target : e.source))
          .filter((n): n is N => n?.fx !== undefined)
        if (linked.length === 0) continue
        // Deterministic offset so the node stays put across reloads.
        const jitter = (edgeShard(node.id, 31) - 15) * 2
        node.fx = node.x = linked.reduce((s, n) => s + n.fx!, 0) / linked.length + jitter
        node.fy = node.y = linked.reduce((s, n) => s + n.fy!, 0) / linked.length - 25
      }
    }
    const links: L[] = edges.map((edge) => ({ source: edge.source, target: edge.target, edge }))
    return { nodes: out, links }
  }, [edges, extraPinned, store, fixedLayout, nodes])

  // The simulation reheats whenever the data changes; camera moves made while
  // it runs are repeated once it settles.
  const engineRunning = useRef(true)
  useEffect(() => {
    engineRunning.current = true
  }, [data])

  useEffect(() => {
    fitPending.current = true
  }, [props.fitKey])

  useEffect(() => {
    const fg = graph.current
    if (!fg || fixedLayout) return
    const charge = fg.d3Force('charge') as unknown as { strength: (s: number) => { distanceMax: (d: number) => void } }
    charge?.strength(-20).distanceMax(200)
    const link = fg.d3Force('link') as unknown as { distance: (d: number) => void }
    link?.distance(24)
  }, [fixedLayout, size.width])

  // Bring the selection into view with its neighbours around it, so a close
  // pair is not blown up to fill the screen.
  const frame = useMemo(() => {
    if (!selection) return new Set<string>(frameIds ?? [])
    const ids = selection.kind === 'disease' ? [selection.id] : [selection.a, selection.b]
    const out = new Set(ids)
    for (const e of edges) {
      if (ids.includes(e.source)) out.add(e.target)
      if (ids.includes(e.target)) out.add(e.source)
    }
    return out
  }, [selection, edges, frameIds])
  // Re-frame when the selection changes or gains edges (e.g. a placement arrives).
  const frameKey = selection ? `${JSON.stringify(selection)}:${frame.size}` : frameIds?.length ? `paper:${frameIds.length}:${frameIds[0]}` : ''
  const frameRef = useRef(frame)
  useEffect(() => {
    frameRef.current = frame
  }, [frame])
  const centerPending = useRef(false)
  const centerOnSelection = useCallback((): boolean => {
    const fg = graph.current
    if (!fg || !frameKey) return false
    const placed = [...frameRef.current].filter((id) => store.get(id)?.x !== undefined)
    if (placed.length === 0) return false
    fg.zoomToFit(600, frameRef.current.size <= 4 ? 220 : 60, (n) => frameRef.current.has(n.id))
    // A handful of close diseases would otherwise be blown up until their dots overlap.
    setTimeout(() => {
      if (fg.zoom() > 3) fg.zoom(3, 300)
    }, 700)
    return true
  }, [frameKey, store])

  useEffect(() => {
    centerPending.current = !centerOnSelection() || engineRunning.current
  }, [centerOnSelection])

  const selectedIds = useMemo(() => {
    if (!selection) return new Set<string>()
    return new Set(selection.kind === 'disease' ? [selection.id] : [selection.a, selection.b])
  }, [selection])

  const drawNode = useCallback(
    (node: N, ctx: CanvasRenderingContext2D, scale: number) => {
      const dimmed = highlightNodes !== null && !highlightNodes.has(node.id)
      // A small highlighted set (a gene's diseases, a paper's claims) is drawn larger so it stands out.
      const picked = highlightNodes !== null && !dimmed && highlightNodes.size <= 300
      const r = nodeRadius(node.degree) * (picked ? 1.5 : 1) + (picked ? 1.5 : 0)
      const selected = selectedIds.has(node.id)
      const hovered = hover?.kind === 'node' && hover.node.id === node.id
      const added = node.disease.origin !== 'catalogue'
      const x = node.x!
      const y = node.y!

      ctx.beginPath()
      if (added) diamond(ctx, x, y, r * 1.4)
      else ctx.arc(x, y, r, 0, 2 * Math.PI)
      const fill = colourOf(node.disease)
      ctx.fillStyle = withAlpha(fill, dimmed ? 0.1 : 0.92)
      ctx.fill()
      if (added && !dimmed) {
        // Added diseases keep a ring so they stay distinguishable from catalogue diseases of the same cluster.
        ctx.lineWidth = 1.5 / scale
        ctx.strokeStyle = colours.ink
        ctx.stroke()
      }

      if (selected || hovered) {
        ctx.beginPath()
        if (added) diamond(ctx, x, y, r * 1.4 + 3 / scale)
        else ctx.arc(x, y, r + 2.5 / scale, 0, 2 * Math.PI)
        ctx.lineWidth = (selected ? 2.5 : 1.5) / scale
        ctx.strokeStyle = colours.selection
        ctx.stroke()
      }

      // Labels for a small set: always when it is tiny, once zoomed in when it would crowd.
      const inFocus =
        highlightNodes !== null &&
        highlightNodes.has(node.id) &&
        (highlightNodes.size <= 3 || (highlightNodes.size <= 60 && scale > 2))
      const showLabel =
        selected ||
        hovered ||
        inFocus ||
        (node.disease.origin === 'user' && !dimmed) ||
        (!dimmed && ((scale > 3 && node.degree >= 25) || scale > 7))
      if (!showLabel) return
      const fontSize = (selected ? 13 : 11) / scale
      ctx.font = `${selected ? 600 : 400} ${fontSize}px system-ui, -apple-system, "Segoe UI", sans-serif`
      ctx.textAlign = 'center'
      // The two ends of a selected pair can sit close together: label the upper one above it.
      const partnerId =
        selection?.kind === 'pair' && selected ? (selection.a === node.id ? selection.b : selection.a) : null
      const partner = partnerId ? store.get(partnerId) : undefined
      const above = partner?.y !== undefined && partner.y > y
      ctx.textBaseline = above ? 'bottom' : 'top'
      const label = displayName(node.disease.name)
      const extent = added ? r * 1.4 : r
      const ly = above ? y - extent - 3 / scale : y + extent + 3 / scale
      ctx.lineJoin = 'round'
      ctx.lineWidth = 3 / scale
      ctx.strokeStyle = withAlpha(colours.surface, 0.9)
      ctx.strokeText(label, x, ly)
      ctx.fillStyle = dimmed ? colours.muted : colours.ink
      ctx.fillText(label, x, ly)
      // Second line: the ORPHA ID and cluster, when the label has room to be read.
      if (selected || hovered || (inFocus && scale > 1.5)) {
        const detail = `${node.disease.id}${node.disease.cluster >= 0 ? ` · ${clusterId(node.disease.cluster)}` : ''}`
        const small = (selected ? 10.5 : 9.5) / scale
        ctx.font = `400 ${small}px system-ui, -apple-system, "Segoe UI", sans-serif`
        const dy = fontSize * 1.15
        const y2 = above ? ly - dy : ly + dy
        ctx.lineWidth = 3 / scale
        ctx.strokeStyle = withAlpha(colours.surface, 0.9)
        ctx.strokeText(detail, x, y2)
        ctx.fillStyle = colours.muted
        ctx.fillText(detail, x, y2)
      }
    },
    [highlightNodes, selectedIds, hover, colours, selection, store, colourOf],
  )

  const paintNodeArea = useCallback((node: N, colour: string, ctx: CanvasRenderingContext2D, scale: number) => {
    // Hit target larger than the dot so small nodes are easy to click.
    ctx.fillStyle = colour
    ctx.beginPath()
    ctx.arc(node.x!, node.y!, nodeRadius(node.degree) + 4 / scale, 0, 2 * Math.PI)
    ctx.fill()
  }, [])

  const linkColour = useCallback(
    (link: L) => {
      const tint = edgeTint?.get(link.edge.id)
      if (tint) return withAlpha(tint, 0.95)
      const base = colours.support[link.edge.support]
      if (hover?.kind === 'link' && hover.link === link) return base
      if (highlightEdges !== null) {
        // A very large highlighted set (every disease of one onset class) is drawn lighter so it stays readable.
        const alpha = highlightEdges.size > 200 ? 0.45 : 0.95
        return highlightEdges.has(link.edge.id) ? withAlpha(base, alpha) : withAlpha(base, 0.05)
      }
      return withAlpha(base, edgeAlpha)
    },
    [colours, highlightEdges, hover, edgeAlpha, edgeTint],
  )

  const linkWidth = useCallback(
    (link: L) => {
      if (edgeTint?.has(link.edge.id)) return hover?.kind === 'link' && hover.link === link ? 3.6 : 2.6
      const emphasised = highlightEdges?.has(link.edge.id) || (hover?.kind === 'link' && hover.link === link)
      return emphasised ? (highlightEdges && highlightEdges.size > 200 ? 1 : 2.2) : 0.7
    },
    [highlightEdges, hover, edgeTint],
  )

  const tooltip = hover && renderTooltip(hover, nodes, clusters, colourOf)
  const tooltipStyle = useMemo(() => {
    const width = 300
    const left = Math.min(pointer.x + 14, size.width - width - 8)
    const top = pointer.y + 14 > size.height - 120 ? pointer.y - 110 : pointer.y + 14
    return { left: Math.max(8, left), top: Math.max(8, top), maxWidth: width }
  }, [pointer, size])

  return (
    <div
      ref={wrapper}
      className="graph-canvas"
      role="img"
      aria-label={`Similarity network of ${data.nodes.length} diseases and ${data.links.length} edges. The table view lists the same edges.`}
      onMouseMove={(e) => {
        const rect = e.currentTarget.getBoundingClientRect()
        setPointer({ x: e.clientX - rect.left, y: e.clientY - rect.top })
      }}
      onMouseLeave={() => setHover(null)}
    >
      {size.width > 0 && (
        <ForceGraph2D<GNode, GLink>
          ref={graph}
          width={size.width}
          height={size.height}
          graphData={data}
          backgroundColor={colours.surface}
          nodeCanvasObject={drawNode}
          nodeCanvasObjectMode={() => 'replace'}
          nodePointerAreaPaint={paintNodeArea}
          linkColor={linkColour}
          linkWidth={linkWidth}
          linkLineDash={(link) => (isClaimOnly(link.edge) ? [5, 4] : link.edge.origin === 'user' ? [2, 2] : null)}
          linkHoverPrecision={4}
          maxZoom={6}
          minZoom={0.05}
          cooldownTicks={fixedLayout ? 0 : 150}
          warmupTicks={fixedLayout ? 0 : 30}
          onEngineStop={() => {
            engineRunning.current = false
            if (centerPending.current && centerOnSelection()) {
              centerPending.current = false
              fitPending.current = false
            } else if (fitPending.current) {
              fitPending.current = false
              graph.current?.zoomToFit(400, 30)
            }
          }}
          onNodeHover={(node) => setHover(node ? { kind: 'node', node } : null)}
          onLinkHover={(link) => setHover(link ? { kind: 'link', link } : null)}
          onNodeClick={(node) => props.onSelectDisease(node.id)}
          onLinkClick={(link) => props.onSelectPair(endId(link.source), endId(link.target))}
          onBackgroundClick={props.onClear}
        />
      )}
      {tooltip && (
        <div className="tooltip" style={tooltipStyle}>
          {tooltip}
        </div>
      )}
    </div>
  )
}

function renderTooltip(
  hover: NonNullable<Hover>,
  nodes: Map<string, GraphNode>,
  clusters: ClusterInfo[],
  colourOf: (d: GraphNode) => string,
) {
  if (hover.kind === 'node') {
    const d = hover.node.disease
    const cluster = clusters[d.cluster]
    return (
      <>
        <div className="tooltip-title">{displayName(d.name)}</div>
        <div className="tooltip-meta">
          {d.origin === 'user' ? 'Added by you' : d.id} · {plural(hover.node.degree, 'edge')}
        </div>
        {d.origin !== 'user' && d.category && <div className="tooltip-meta">{d.category}</div>}
        {cluster && (
          <div className="tooltip-meta">
            <span className="swatch" style={{ background: colourOf(d) }} aria-hidden />
            {cluster.id} · {cluster.label}
            {d.origin === 'user' ? ' (nearest neighbours)' : ''}
          </div>
        )}
      </>
    )
  }
  const e = hover.link.edge
  if (isClaimOnly(e)) {
    return (
      <>
        <div className="tooltip-title">
          {displayName(nodes.get(e.source)?.name ?? e.source)} – {displayName(nodes.get(e.target)?.name ?? e.target)}
        </div>
        <div className="tooltip-meta">A paper says these are similar; the graph has no edge between them.</div>
      </>
    )
  }
  return (
    <>
      <div className="tooltip-value">
        {percentile(e.percentile)} <span className="tooltip-meta">percentile vs random pairs</span>
      </div>
      <div className="tooltip-title">
        {displayName(nodes.get(e.source)?.name ?? e.source)} – {displayName(nodes.get(e.target)?.name ?? e.target)}
      </div>
      <div className="tooltip-meta">
        <span className={`line-key support-${e.support}${e.origin === 'user' ? ' is-user' : ''}`} aria-hidden />
        {e.support} · mainly {modalityLabel(e.mainModality).toLowerCase()}
      </div>
    </>
  )
}
