import { useMemo, useRef, useState } from 'react'
import type { PaperEntry } from '../data/types'
import { ACCEPTED_FILES } from '../lib/paperService'
import { type ClaimEvaluation, countVerdicts, recall, VERDICT_ORDER } from '../lib/paperEval'
import { verdictColour } from '../lib/theme'
import { plural } from '../lib/format'
import type { UserDisease } from '../lib/userDiseases'

interface Props {
  uploads: UserDisease[]
  literature: PaperEntry[] | null
  literatureError: string | null
  literatureEval: Map<string, ClaimEvaluation[]> | null
  activeId: string | null
  /** Stage text while an upload is being processed. */
  busy: string | null
  error: string | null
  dark: boolean
  onUpload: (file: File) => void
  onActivate: (id: string | null) => void
  onRemoveUpload: (disease: UserDisease) => void
}

const LISTED = 30

function VerdictDots({ evaluations, dark }: { evaluations: ClaimEvaluation[]; dark: boolean }) {
  return (
    <span className="verdict-dots" aria-hidden>
      {evaluations.slice(0, 8).map((e, i) => (
        <span key={i} style={{ background: verdictColour(e.verdict ?? 'missing', dark) }} />
      ))}
    </span>
  )
}

/** Upload a paper, and browse the literature set; choosing one overlays its claims on the graph. */
export function PapersPanel(props: Props) {
  const { uploads, literature, literatureEval, activeId, dark } = props
  const file = useRef<HTMLInputElement>(null)
  const [over, setOver] = useState(false)
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState(false)

  const summary = useMemo(() => {
    if (!literatureEval) return null
    const all = [...literatureEval.values()].flat()
    const counts = countVerdicts(all)
    const checkable = counts.confirmed + counts.partial + counts.unsupported
    return { papers: literatureEval.size, claims: all.length, counts, recall: recall(counts), checkable }
  }, [literatureEval])

  const shown = useMemo(() => {
    if (!literature) return []
    const q = query.trim().toLowerCase()
    const list = q ? literature.filter((p) => p.title.toLowerCase().includes(q)) : literature
    return list.slice(0, LISTED)
  }, [literature, query])

  const take = (files: FileList | null) => {
    const f = files?.[0]
    if (f) props.onUpload(f)
    if (file.current) file.current.value = ''
  }

  return (
    <div className="papers-panel">
      <div className="filters-header">
        <span className="section-label">Papers</span>
        {activeId && (
          <button type="button" className="link-button small" onClick={() => props.onActivate(null)}>
            Hide paper edges
          </button>
        )}
      </div>

      <div
        className={`dropzone ${over ? 'is-over' : ''}`}
        onDragOver={(e) => {
          e.preventDefault()
          setOver(true)
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault()
          setOver(false)
          take(e.dataTransfer.files)
        }}
      >
        <input ref={file} type="file" hidden accept={ACCEPTED_FILES} onChange={(e) => take(e.target.files)} />
        <button type="button" className="button full" disabled={props.busy !== null} onClick={() => file.current?.click()}>
          {props.busy ? `${props.busy}…` : 'Upload a paper'}
        </button>
        <p className="muted small">
          PDF, text, XML, or a v2_5 result (JSON). Extraction is a <strong>mock</strong> until the v2_5 pipeline is connected.
        </p>
      </div>
      {props.error && <p className="callout small" role="alert">{props.error}</p>}

      {uploads.length > 0 && (
        <ul className="paper-list">
          {uploads.map((d) => (
            <li key={d.id} className={d.paper!.id === activeId ? 'is-selected' : undefined}>
              <button
                type="button"
                className="paper-item"
                onClick={() => props.onActivate(d.paper!.id === activeId ? null : d.paper!.id)}
              >
                <span className="paper-title">{d.paper!.title}</span>
                <span className="muted small">
                  {d.paper!.mock && <span className="badge-mock">mock</span>} {plural(d.paper!.claims.length, 'claim')}
                </span>
              </button>
              <button type="button" className="link-button small" onClick={() => props.onRemoveUpload(d)}>
                Remove
              </button>
            </li>
          ))}
        </ul>
      )}

      <button type="button" className="disclosure" aria-expanded={open} onClick={() => setOpen(!open)}>
        <span aria-hidden>{open ? '▾' : '▸'}</span> Literature set
        {summary && <span className="muted small"> · {plural(summary.papers, 'paper')}</span>}
      </button>
      {open && (
        <>
          {props.literatureError && <p className="callout small">{props.literatureError}</p>}
          {!literature && !props.literatureError && <p className="muted small">Loading…</p>}
          {summary && (
            <div className="lit-summary small">
              <p>
                <strong>{Math.round(summary.recall * 100)}%</strong> of {plural(summary.claims, 'claim')} have an edge in the
                graph;{' '}
                <strong>
                  {summary.checkable > 0 ? Math.round((summary.counts.confirmed / summary.checkable) * 100) : 0}%
                </strong>{' '}
                of the {summary.checkable} with a checkable dimension are fully carried.
              </p>
              <p className="muted">
                The pairs were chosen for a paper-first benchmark, partly by phenotype similarity, so this overstates how well
                the graph would recover arbitrary claims.
              </p>
              <div className="verdict-bar" role="img" aria-label="Share of claims by verdict">
                {VERDICT_ORDER.map((v) =>
                  summary.counts[v] > 0 ? (
                    <span
                      key={v}
                      title={`${v}: ${summary.counts[v]}`}
                      style={{ flex: summary.counts[v], background: verdictColour(v, dark) }}
                    />
                  ) : null,
                )}
              </div>
            </div>
          )}
          <input
            type="search"
            className="text-input"
            placeholder="Search paper titles"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            aria-label="Search literature papers"
          />
          <ul className="paper-list">
            {shown.map((p) => (
              <li key={p.id} className={p.id === activeId ? 'is-selected' : undefined}>
                <button type="button" className="paper-item" onClick={() => props.onActivate(p.id === activeId ? null : p.id)}>
                  <span className="paper-title">{p.title}</span>
                  <span className="muted small">
                    {p.year ?? ''} {plural(p.claims.length, 'claim')}{' '}
                    {literatureEval?.get(p.id) && <VerdictDots evaluations={literatureEval.get(p.id)!} dark={dark} />}
                  </span>
                </button>
              </li>
            ))}
          </ul>
          {literature && literature.length > shown.length && (
            <p className="muted small">
              Showing {shown.length} of {literature.length}. Search to narrow the list.
            </p>
          )}
        </>
      )}
    </div>
  )
}
