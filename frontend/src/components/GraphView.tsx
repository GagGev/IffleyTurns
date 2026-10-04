import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import ForceGraph2D from 'react-force-graph-2d'
import type { ForceGraphMethods, LinkObject, NodeObject } from 'react-force-graph-2d'
import type { ComputedEdge, Disease } from '../data/types'
import type { EdgeView } from '../lib/graph'
import { displayName, plural, score } from '../lib/format'
import type { Selection } from '../lib/selection'
import { type CanvasColours, withAlpha } from '../lib/theme'

interface GNode {
  id: string
  disease: Disease
  degree: number
}
/** A literature edge (`view`) or a feature-based link from an added disease (`computed`). */
interface GLink {
  view?: EdgeView
  computed?: ComputedEdge
}
type N = NodeObject<GNode>
type L = LinkObject<GNode, GLink>

interface Props {
  diseases: Map<string, Disease>
  edges: EdgeView[]
  computedEdges: ComputedEdge[]
  /** Diseases to draw even without visible edges (the current selection). */
  pinnedIds: string[]
  selection: Selection
  /** Diseases to keep at full strength; everything else is dimmed. */
  highlightNodes: Set<string> | null
  highlightEdges: Set<string> | null
  measureLabel: string
  /** Changes whenever the graph should re-fit to the viewport. */
  fitKey: string
  colours: CanvasColours
  onSelectDisease: (id: string) => void
  onSelectPair: (a: string, b: string) => void
  onClear: () => void
}

type Hover = { kind: 'node'; node: N } | { kind: 'link'; link: L } | null

/**
 * Layout state owned by the force simulation, which writes x/y into these
 * node objects.  Kept outside React state on purpose: reusing the objects is
 * what keeps positions stable when filters change.
 */
class NodeStore {
  private nodes = new Map<string, N>()
  private diseases = new Map<string, Disease>()

  sync(diseases: Map<string, Disease>) {
    if (diseases === this.diseases) return
    this.diseases = diseases
    for (const [id, node] of this.nodes) {
      const d = diseases.get(id)
      if (d) node.disease = d
    }
  }

  node(id: string, degree: number): N | undefined {
    const disease = this.diseases.get(id)
    if (!disease) return undefined
    let node = this.nodes.get(id)
    if (!node) {
      node = { id, disease, degree }
      this.nodes.set(id, node)
    }
    node.degree = degree
    return node
  }

  get(id: string): N | undefined {
    return this.nodes.get(id)
  }
}

const nodeRadius = (degree: number) => 2.5 + Math.sqrt(degree) * 1.6
const linkId = (link: L) => (link.computed ? link.computed.id : link.view!.edge.id)
const endId = (end: L['source']) => (typeof end === 'object' ? (end as N).id : String(end))

export function GraphView(props: Props) {
  const { diseases, edges, computedEdges, pinnedIds, selection, highlightNodes, highlightEdges, colours } = props
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
  // data; otherwise selecting a node would restart the simulation.
  const extraPinned = useMemo(() => {
    const drawn = new Set<string>()
    for (const { source, target } of [...edges.map((v) => v.edge), ...computedEdges]) {
      drawn.add(source)
      drawn.add(target)
    }
    return pinnedIds.filter((id) => !drawn.has(id)).join('\n')
  }, [edges, computedEdges, pinnedIds])

  // Node objects are reused across filter changes so the layout stays put.
  // Disease records come from the store, so a new disease map (e.g. after a
  // selection) does not rebuild the graph data and restart the simulation.
  store.sync(diseases)
  const data = useMemo(() => {
    const degree = new Map<string, number>()
    for (const { source, target } of [...edges.map((v) => v.edge), ...computedEdges]) {
      degree.set(source, (degree.get(source) ?? 0) + 1)
      degree.set(target, (degree.get(target) ?? 0) + 1)
    }
    for (const id of extraPinned ? extraPinned.split('\n') : []) degree.set(id, 0)
    const nodes: N[] = []
    for (const [id, d] of degree) {
      const node = store.node(id, d)
      if (node) nodes.push(node)
    }
    const links: L[] = [
      ...edges.map((view) => ({ source: view.edge.source, target: view.edge.target, view })),
      ...computedEdges.map((computed) => ({ source: computed.source, target: computed.target, computed })),
    ]
    return { nodes, links }
  }, [edges, computedEdges, extraPinned, store])

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
    if (!fg) return
    // Short-range repulsion keeps the many small components near the centre.
    const charge = fg.d3Force('charge') as unknown as { strength: (s: number) => { distanceMax: (d: number) => void } }
    charge?.strength(-28).distanceMax(220)
    const link = fg.d3Force('link') as unknown as { distance: (d: number) => void }
    link?.distance(28)
  }, [])

  // Bring the selection (and, for a disease, its neighbours) into view, or wait
  // for the layout if those nodes have no position yet.
  const centerPending = useRef(false)
  const focusRef = useRef(highlightNodes)
  useEffect(() => {
    focusRef.current = highlightNodes
  }, [highlightNodes])
  const centerOnSelection = useCallback((): boolean => {
    const fg = graph.current
    if (!fg || !selection) return false
    const ids = selection.kind === 'disease' ? [selection.id] : [selection.a, selection.b]
    if (!ids.some((id) => store.get(id)?.x !== undefined)) return false
    const frame = selection.kind === 'disease' && focusRef.current ? focusRef.current : new Set(ids)
    // One camera move only: separate centerAt + zoom transitions interrupt each other.
    fg.zoomToFit(600, 120, (n) => frame.has(n.id))
    return true
  }, [selection, store])

  useEffect(() => {
    centerPending.current = !centerOnSelection() || engineRunning.current
  }, [centerOnSelection])

  const selectedIds = useMemo(() => {
    if (!selection) return new Set<string>()
    return new Set(selection.kind === 'disease' ? [selection.id] : [selection.a, selection.b])
  }, [selection])

  const drawNode = useCallback(
    (node: N, ctx: CanvasRenderingContext2D, scale: number) => {
      const r = nodeRadius(node.degree)
      const dimmed = highlightNodes !== null && !highlightNodes.has(node.id)
      const selected = selectedIds.has(node.id)
      const hovered = hover?.kind === 'node' && hover.node.id === node.id
      const alpha = dimmed ? 0.15 : 1
      const common = node.disease.status === 'common'

      const user = node.disease.origin === 'user'
      ctx.beginPath()
      if (user) {
        // Added diseases are diamonds, so they are identifiable without colour.
        const d = r * 1.35
        ctx.moveTo(node.x!, node.y! - d)
        ctx.lineTo(node.x! + d, node.y!)
        ctx.lineTo(node.x!, node.y! + d)
        ctx.lineTo(node.x! - d, node.y!)
        ctx.closePath()
      } else {
        ctx.arc(node.x!, node.y!, r, 0, 2 * Math.PI)
      }
      if (common) {
        ctx.fillStyle = withAlpha(colours.surface, alpha)
        ctx.fill()
        ctx.lineWidth = Math.max(1.4 / scale, 0.6)
        ctx.strokeStyle = withAlpha(colours.node, alpha)
        ctx.stroke()
      } else {
        ctx.fillStyle = withAlpha(colours.node, alpha)
        ctx.fill()
      }
      if (selected || hovered) {
        ctx.beginPath()
        if (user) {
          const d = r * 1.35 + 3.5 / scale
          ctx.moveTo(node.x!, node.y! - d)
          ctx.lineTo(node.x! + d, node.y!)
          ctx.lineTo(node.x!, node.y! + d)
          ctx.lineTo(node.x! - d, node.y!)
          ctx.closePath()
        } else {
          ctx.arc(node.x!, node.y!, r + 2.5 / scale, 0, 2 * Math.PI)
        }
        ctx.lineWidth = (selected ? 2.5 : 1.5) / scale
        ctx.strokeStyle = colours.selection
        ctx.stroke()
      }

      const inFocus = highlightNodes !== null && highlightNodes.has(node.id) && highlightNodes.size <= 40
      const showLabel =
        selected || hovered || inFocus || (user && !dimmed) || (!dimmed && ((scale > 2.4 && node.degree >= 3) || scale > 4.5))
      if (!showLabel) return
      const fontSize = (selected ? 13 : 11) / scale
      ctx.font = `${selected ? 600 : 400} ${fontSize}px system-ui, -apple-system, "Segoe UI", sans-serif`
      ctx.textAlign = 'center'
      // The two ends of a selected pair sit close together: label the upper one above it.
      const partnerId =
        selection?.kind === 'pair' && selected ? (selection.a === node.id ? selection.b : selection.a) : null
      const partner = partnerId ? store.get(partnerId) : undefined
      const above = partner?.y !== undefined && partner.y > node.y!
      ctx.textBaseline = above ? 'bottom' : 'top'
      const label = displayName(node.disease.name)
      const extent = user ? r * 1.35 : r
      const y = above ? node.y! - extent - 3 / scale : node.y! + extent + 3 / scale
      ctx.lineJoin = 'round'
      ctx.lineWidth = 3 / scale
      ctx.strokeStyle = withAlpha(colours.surface, 0.9)
      ctx.strokeText(label, node.x!, y)
      ctx.fillStyle = dimmed ? colours.muted : colours.ink
      ctx.fillText(label, node.x!, y)
    },
    [highlightNodes, selectedIds, hover, colours, selection, store],
  )

  const paintNodeArea = useCallback((node: N, colour: string, ctx: CanvasRenderingContext2D, scale: number) => {
    // Hit target larger than the dot so small nodes are easy to click.
    ctx.fillStyle = colour
    ctx.beginPath()
    ctx.arc(node.x!, node.y!, nodeRadius(node.degree) + 5 / scale, 0, 2 * Math.PI)
    ctx.fill()
  }, [])

  const linkColour = useCallback(
    (link: L) => {
      const base = link.computed ? colours.computed : colours.relationship[link.view!.agg.relationship]
      const value = link.computed ? link.computed.similarity : link.view!.value
      const isHover = hover?.kind === 'link' && hover.link === link
      if (isHover) return base
      if (highlightEdges !== null && !highlightEdges.has(linkId(link))) return withAlpha(base, 0.06)
      return withAlpha(base, 0.3 + 0.65 * value)
    },
    [colours, highlightEdges, hover],
  )

  const linkWidth = useCallback(
    (link: L) => {
      const base = link.computed
        ? 1 + link.computed.similarity * 2.5
        : 0.6 + Math.min(link.view!.agg.nPapers, 6) * 0.45
      const emphasised = highlightEdges?.has(linkId(link)) || (hover?.kind === 'link' && hover.link === link)
      return emphasised ? base * 1.7 : base
    },
    [highlightEdges, hover],
  )

  const tooltip = hover && renderTooltip(hover, diseases, props.measureLabel)
  const tooltipStyle = useMemo(() => {
    const width = 280
    const left = Math.min(pointer.x + 14, size.width - width - 8)
    const top = pointer.y + 14 > size.height - 120 ? pointer.y - 110 : pointer.y + 14
    return { left: Math.max(8, left), top: Math.max(8, top), maxWidth: width }
  }, [pointer, size])

  return (
    <div
      ref={wrapper}
      className="graph-canvas"
      role="img"
      aria-label={`Network of ${data.nodes.length} diseases and ${data.links.length} literature-backed pairs. The table view lists the same pairs.`}
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
          linkLineDash={(link) =>
            link.computed ? [1.5, 2.5] : link.view!.agg.relationship === 'unrelated' ? [3, 2] : null
          }
          linkHoverPrecision={6}
          maxZoom={5}
          cooldownTicks={200}
          warmupTicks={40}
          onEngineStop={() => {
            engineRunning.current = false
            if (centerPending.current && centerOnSelection()) {
              centerPending.current = false
              fitPending.current = false
            } else if (fitPending.current) {
              fitPending.current = false
              graph.current?.zoomToFit(400, 40)
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

function renderTooltip(hover: NonNullable<Hover>, diseases: Map<string, Disease>, measureLabel: string) {
  if (hover.kind === 'node') {
    const d = hover.node.disease
    return (
      <>
        <div className="tooltip-title">{displayName(d.name)}</div>
        <div className="tooltip-meta">
          {plural(hover.node.degree, 'connection')} · {d.status}
          {d.orphaId ? ` · ${d.orphaId}` : ' · no ORPHA ID'}
        </div>
      </>
    )
  }
  if (hover.link.computed) {
    const c = hover.link.computed
    return (
      <>
        <div className="tooltip-value">
          {score(c.similarity)} <span className="tooltip-meta">feature similarity</span>
        </div>
        <div className="tooltip-title">
          {displayName(diseases.get(c.source)?.name ?? c.source)} – {displayName(diseases.get(c.target)?.name ?? c.target)}
        </div>
        <div className="tooltip-meta">
          <span className="line-key rel-computed" aria-hidden />
          computed from shared features
        </div>
      </>
    )
  }
  const { edge, agg, value } = hover.link.view!
  const a = diseases.get(edge.source)
  const b = diseases.get(edge.target)
  return (
    <>
      <div className="tooltip-value">
        {score(value)} <span className="tooltip-meta">{measureLabel.toLowerCase()}</span>
      </div>
      <div className="tooltip-title">
        {displayName(a?.name ?? edge.source)} – {displayName(b?.name ?? edge.target)}
      </div>
      <div className="tooltip-meta">
        <span className={`line-key rel-${agg.relationship.replace(/ /g, '-')}`} aria-hidden />
        {agg.relationship} · {plural(agg.nPapers, 'paper')} · evidence {agg.evidenceScore}/15
      </div>
    </>
  )
}
