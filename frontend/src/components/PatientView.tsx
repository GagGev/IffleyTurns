import { useEffect, useMemo, useState } from 'react'
import { loadAnnotations, type TermOption } from '../data/annotations'
import {
  explainDisease,
  explainGroup,
  groupInfo,
  interpretSymptoms,
  type InterpretedTerm,
  type KeyPaper,
  keyPapers,
  patientStatus,
  RESOURCES,
} from '../data/patient'
import { apiHealth, placeDisease } from '../data/source'
import type { ApiHealth, GraphData, GraphNode, PlacementNeighbour } from '../data/types'
import { displayName, orphanetUrl } from '../lib/format'
import type { AppView } from '../lib/route'
import { TermPicker } from './TermPicker'
import { ThemeToggle } from './ThemeToggle'

interface Symptom {
  id: string
  label: string
}

interface Example {
  id: string
  name: string
  /** Symptoms the condition shares with the ones entered. */
  shared: string[]
}

interface Group {
  index: number
  category: string
  examples: Example[]
}

const NEIGHBOURS = 30
const GROUPS_SHOWN = 3
const EXAMPLES_SHOWN = 4

/**
 * Rank groups by how many of the closest diseases fall in them, closer ones
 * counting more.  Clusters are merged by their main Orphanet category, since
 * that is what the plain-language group name is based on.
 */
function groupNeighbours(neighbours: PlacementNeighbour[], nodes: Map<string, GraphNode>, graph: GraphData): Group[] {
  const groups = new Map<string, { score: number; index: number; examples: Example[] }>()
  neighbours.forEach((n, rank) => {
    const node = nodes.get(n.id)
    const category = node && node.cluster >= 0 ? graph.clusters[node.cluster]?.topCategory : undefined
    if (!node || !category) return
    const g = groups.get(category) ?? { score: 0, index: node.cluster, examples: [] }
    g.score += 1 / (rank + 1)
    // Patients don't need the HPO codes v2 appends to feature names.
    const shared = (n.explanation.find((e) => e.modality === 'phenotype')?.shared ?? []).map((f) =>
      f.replace(/\s*\(HP:\d+\)$/, ''),
    )
    g.examples.push({ id: n.id, name: n.name, shared })
    groups.set(category, g)
  })
  return [...groups.entries()]
    .sort((a, b) => b[1].score - a[1].score)
    .slice(0, GROUPS_SHOWN)
    .map(([category, g]) => ({ index: g.index, category, examples: g.examples.slice(0, EXAMPLES_SHOWN) }))
}

export function PatientView({ graph, onNavigate }: { graph: GraphData; onNavigate: (view: AppView) => void }) {
  const nodes = useMemo(() => new Map(graph.nodes.map((n) => [n.id, n])), [graph])
  const [medgemma, setMedgemma] = useState<'checking' | 'ready' | 'offline'>('checking')
  const [placement, setPlacement] = useState<ApiHealth>({ status: 'loading' })
  const [options, setOptions] = useState<TermOption[] | null>(null)

  const [text, setText] = useState('')
  const [interpreting, setInterpreting] = useState(false)
  const [interpreted, setInterpreted] = useState<InterpretedTerm[] | null>(null)
  const [interpretError, setInterpretError] = useState<string | null>(null)
  const [symptoms, setSymptoms] = useState<Symptom[]>([])

  const [matching, setMatching] = useState(false)
  const [matchError, setMatchError] = useState<string | null>(null)
  const [groups, setGroups] = useState<Group[] | null>(null)

  // Keep checking until both services are up, so starting them after the page loads still works.
  useEffect(() => {
    let cancelled = false
    let timer: ReturnType<typeof setTimeout>
    const check = async () => {
      const [m, p] = await Promise.all([patientStatus(), apiHealth()])
      if (cancelled) return
      setMedgemma(m)
      setPlacement(p)
      if (m !== 'ready' || p.status !== 'ready') timer = setTimeout(check, p.status === 'loading' ? 2000 : 5000)
    }
    void check()
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [])

  useEffect(() => {
    loadAnnotations().then(
      (a) => setOptions(a.symptoms),
      () => setOptions([]),
    )
  }, [])

  const has = (id: string) => symptoms.some((s) => s.id === id)
  const add = (s: Symptom) => setSymptoms((list) => (list.some((x) => x.id === s.id) ? list : [...list, s]))
  const remove = (id: string) => setSymptoms((list) => list.filter((s) => s.id !== id))
  const labelOf = (id: string) => options?.find((o) => o.id === id)?.label ?? id

  const interpret = async () => {
    setInterpreting(true)
    setInterpretError(null)
    try {
      const { terms } = await interpretSymptoms(text)
      setInterpreted(terms)
      // Add the best match for each term; the patient can untick any that are wrong.
      for (const t of terms) if (t.matches[0]) add(t.matches[0])
    } catch (e) {
      setInterpretError((e as Error).message)
    } finally {
      setInterpreting(false)
    }
  }

  const findGroups = async () => {
    setMatching(true)
    setMatchError(null)
    try {
      const placement = await placeDisease(
        'USER:patient',
        { name: 'Symptoms entered', phenotypes: Object.fromEntries(symptoms.map((s) => [s.id, 1])) },
        [],
        NEIGHBOURS,
      )
      setGroups(groupNeighbours(placement.neighbours, nodes, graph))
      window.scrollTo({ top: 0, behavior: 'smooth' })
    } catch (e) {
      setMatchError((e as Error).message)
    } finally {
      setMatching(false)
    }
  }

  const startAgain = () => {
    setGroups(null)
    setInterpreted(null)
    setSymptoms([])
    setText('')
  }

  return (
    <div className="patient">
      <header className="patient-header">
        <div className="patient-header-inner">
          <button type="button" className="link-button" onClick={() => onNavigate('home')}>
            ← Home
          </button>
          <span className="patient-brand">Rare Disease Explorer · for patients and families</span>
          <span className="header-nav">
            <button type="button" className="link-button" onClick={() => onNavigate('research')}>
              Researcher view
            </button>
            <ThemeToggle />
          </span>
        </div>
      </header>

      <main className="patient-main">
        <p className="patient-notice" role="note">
          <strong>Not a diagnosis.</strong> This shows groups of rare conditions that share some of the symptoms you
          choose. Many common conditions cause the same symptoms, so please talk to a doctor about any worries. If
          symptoms are severe or getting worse quickly, contact your doctor or emergency services.
        </p>

        {groups ? (
          <Results groups={groups} symptoms={symptoms} medgemma={medgemma === 'ready'} onStartAgain={startAgain} onEdit={() => setGroups(null)} />
        ) : (
          <section className="patient-card">
            <h2>Choose symptoms</h2>
            <TermPicker
              label="Search for a symptom"
              placeholder="e.g. seizure, tall stature, easy bruising"
              options={(options ?? []).filter((o) => !has(o.id))}
              value={null}
              loading={options === null}
              onPick={(id) => add({ id, label: labelOf(id) })}
              onClear={() => {}}
            />
            {symptoms.length > 0 ? (
              <ul className="patient-symptoms" aria-label="Chosen symptoms">
                {symptoms.map((s) => (
                  <li key={s.id}>
                    {s.label}
                    <button type="button" aria-label={`Remove ${s.label}`} onClick={() => remove(s.id)}>
                      ×
                    </button>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="muted small">No symptoms chosen yet.</p>
            )}

            {medgemma === 'ready' && (
              <div className="describe">
                <label htmlFor="patient-text" className="patient-label">
                  Or describe them in your own words
                </label>
                <textarea
                  id="patient-text"
                  rows={3}
                  value={text}
                  maxLength={2000}
                  placeholder="For example: my son has fits, is very floppy, and can't sit up yet at one year old."
                  onChange={(e) => setText(e.target.value)}
                />
                <button type="button" className="button" disabled={!text.trim() || interpreting} onClick={() => void interpret()}>
                  {interpreting ? 'Reading your description…' : 'Find the symptoms'}
                </button>
                <p className="muted small">Processed on this computer and not saved.</p>
                {interpretError && <p className="callout small">{interpretError}</p>}
                  {interpreted && (
                    <div className="interpreted">
                      <h3>What we understood</h3>
                      {interpreted.length === 0 ? (
                        <p className="muted">We couldn't find any symptoms in that description. Try describing them differently, or search above.</p>
                      ) : (
                        <ul className="interpreted-list">
                          {interpreted.map((t) => (
                            <li key={t.term}>
                              {t.matches.length > 0 ? (
                                <>
                                  <label className="check">
                                    <input
                                      type="checkbox"
                                      checked={t.matches.some((m) => has(m.id))}
                                      onChange={(e) => {
                                        const current = t.matches.find((m) => has(m.id))
                                        if (e.target.checked) add(t.matches[0])
                                        else if (current) remove(current.id)
                                      }}
                                    />
                                    {t.matches.length === 1 && <span>{t.matches[0].label}</span>}
                                    {t.matches.length > 1 && !t.matches.some((m) => has(m.id)) && (
                                      <span className="muted">{t.matches[0].label}</span>
                                    )}
                                  </label>
                                  {t.matches.length > 1 && t.matches.some((m) => has(m.id)) && (
                                    <select
                                      aria-label={`Other meanings of ${t.term}`}
                                      value={t.matches.find((m) => has(m.id))?.id ?? ''}
                                      onChange={(e) => {
                                        for (const m of t.matches) remove(m.id)
                                        const pick = t.matches.find((m) => m.id === e.target.value)
                                        if (pick) add(pick)
                                      }}
                                    >
                                      <option value="">Not this</option>
                                      {t.matches.map((m) => (
                                        <option key={m.id} value={m.id}>
                                          {m.label}
                                        </option>
                                      ))}
                                    </select>
                                  )}
                                </>
                              ) : (
                                <span className="muted">
                                  We couldn't match “{t.term}” to a medical term. Try searching for it above.
                                </span>
                              )}
                            </li>
                          ))}
                        </ul>
                      )}
                      <p className="muted small">Please check these. Untick anything that isn't right.</p>
                    </div>
                  )}
              </div>
            )}

            <div className="patient-actions">
              <button
                type="button"
                className="button primary"
                disabled={symptoms.length === 0 || matching || placement.status !== 'ready'}
                onClick={() => void findGroups()}
              >
                {matching ? 'Looking…' : 'Show related groups'}
              </button>
              {placement.status !== 'ready' && (
                <span className="muted small">
                  {placement.status === 'loading'
                    ? 'Getting ready, this takes a few seconds…'
                    : "Results can't be shown right now: the matching service isn't running."}
                </span>
              )}
            </div>
            {matchError && <p className="callout small">{matchError}</p>}
          </section>
        )}
      </main>
    </div>
  )
}

function Results({
  groups,
  symptoms,
  medgemma,
  onStartAgain,
  onEdit,
}: {
  groups: Group[]
  symptoms: Symptom[]
  medgemma: boolean
  onStartAgain: () => void
  onEdit: () => void
}) {
  return (
    <>
      <section className="patient-card">
        <h2>Groups of conditions that share these symptoms</h2>
        <p>
          Based on: <strong>{symptoms.map((s) => s.label).join(', ')}</strong>.{' '}
          <button type="button" className="link-button" onClick={onEdit}>
            Change symptoms
          </button>
        </p>
        <p className="muted small">
          These are the groups whose conditions most resemble the symptoms entered, closest first. Most people with these
          symptoms do not have a rare condition.
        </p>
      </section>
      {groups.length === 0 && (
        <section className="patient-card">
          <p>No related groups were found for these symptoms. Try adding more detail.</p>
        </section>
      )}
      {groups.map((g, i) => (
        <GroupCard key={g.category} group={g} first={i === 0} medgemma={medgemma} />
      ))}
      <section className="patient-card">
        <h2>Where to find reliable help</h2>
        <ul className="resource-list">
          <li>
            <strong>Your GP or family doctor</strong> is the best first step. Take a list of the symptoms, when they
            started and any family history. They can refer you to a specialist or to clinical genetics.
          </li>
          {RESOURCES.map((r) => (
            <li key={r.url}>
              <a href={r.url} target="_blank" rel="noreferrer">
                {r.name}
              </a>
              : {r.about}
            </li>
          ))}
        </ul>
      </section>
      <div className="patient-end">
        <button type="button" className="button" onClick={onStartAgain}>
          Start again
        </button>
      </div>
    </>
  )
}

function GroupCard({ group, first, medgemma }: { group: Group; first: boolean; medgemma: boolean }) {
  const info = groupInfo(group.category)
  const [summary, setSummary] = useState<string | null>(null)
  const shared = useMemo(() => [...new Set(group.examples.flatMap((e) => e.shared))].slice(0, 8), [group])

  useEffect(() => {
    if (!medgemma) return
    let cancelled = false
    explainGroup(info.name, group.examples.map((e) => e.name), shared).then(
      (r) => !cancelled && setSummary(r.text),
      () => {},
    )
    return () => {
      cancelled = true
    }
  }, [medgemma, info.name, group, shared])

  return (
    <section className="patient-card group-card">
      {first && <span className="badge">Closest group</span>}
      <h2>{info.name}</h2>
      <p>{summary ?? info.about}</p>
      {medgemma && !summary && <p className="muted small">Writing a simple explanation…</p>}

      <h3>Conditions in this group with similar features</h3>
      <ul className="example-grid">
        {group.examples.map((e) => (
          <ExampleCard key={e.id} example={e} medgemma={medgemma} />
        ))}
      </ul>

      <h3>Who to talk to</h3>
      <p>
        Doctors who usually look after these conditions: <strong>{info.doctors}</strong>. Your GP can refer you if they
        think it is needed.
      </p>
    </section>
  )
}

function ExampleCard({ example, medgemma }: { example: Example; medgemma: boolean }) {
  const [open, setOpen] = useState(false)
  // undefined: not asked yet; null: Orphanet has no description to simplify.
  const [text, setText] = useState<string | null | undefined>(undefined)
  const [error, setError] = useState<string | null>(null)
  const [papers, setPapers] = useState<{ list: KeyPaper[]; searchUrl: string } | null>(null)

  useEffect(() => {
    let cancelled = false
    keyPapers(example.id).then(
      (r) => !cancelled && setPapers({ list: r.papers, searchUrl: r.searchUrl }),
      () => {},
    )
    return () => {
      cancelled = true
    }
  }, [example.id])

  const toggle = () => {
    setOpen(!open)
    if (!open && text === undefined && medgemma) {
      explainDisease(example.id, example.name).then(
        (r) => setText(r.text),
        (e: Error) => setError(e.message),
      )
    }
  }

  return (
    <li className="example-card">
      <h4>{displayName(example.name)}</h4>
      {example.shared.length > 0 && (
        <ul className="shared-tags" aria-label="Symptoms it shares">
          {example.shared.map((s) => (
            <li key={s}>{s}</li>
          ))}
        </ul>
      )}
      <div className="example-links">
        <a href={orphanetUrl(example.id)} target="_blank" rel="noreferrer">
          Orphanet page
        </a>
        {medgemma && (
          <button type="button" className="link-button" aria-expanded={open} onClick={toggle}>
            {open ? 'Hide explanation' : 'Explain simply'}
          </button>
        )}
      </div>
      {open && (
        <p className="example-explanation">
          {error ??
            (text === undefined
              ? 'Writing a simple explanation…'
              : text === null
                ? 'There is no summary of this condition to simplify yet. Its Orphanet page may have more information.'
                : text)}
          {text && <span className="muted small"> (Simplified from Orphanet's description by an AI model; may contain mistakes.)</span>}
        </p>
      )}
      <div className="key-papers">
        <span className="key-papers-label">Key papers</span>
        {papers === null ? (
          <span className="muted small">Looking…</span>
        ) : papers.list.length > 0 ? (
          <ul>
            {papers.list.map((p) => (
              <li key={p.url}>
                <a href={p.url} target="_blank" rel="noreferrer" title={p.title}>
                  {p.title}
                </a>
                <span className="muted small">
                  {[p.journal, p.year].filter(Boolean).join(', ')}
                  {p.openAccess && ' · free to read'}
                </span>
              </li>
            ))}
          </ul>
        ) : (
          <a className="small" href={papers.searchUrl} target="_blank" rel="noreferrer">
            Search Europe PMC for papers
          </a>
        )}
      </div>
    </li>
  )
}
