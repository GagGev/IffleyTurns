import { useEffect, useId, useMemo, useRef, useState } from 'react'
import { placeDisease } from '../data/source'
import type { ApiHealth, DiseaseInput, Placement } from '../data/types'
import { displayName, download, modalityLabel, percentile, plural } from '../lib/format'
import {
  compactInput,
  countFeatures,
  type ImportItem,
  INHERITANCE_OPTIONS,
  JSON_TEMPLATE,
  normaliseHpo,
  ONSET_OPTIONS,
  parseImport,
  phenotypeWeights,
  PREVALENCE_CLASSES,
  type UserDisease,
} from '../lib/userDiseases'
import { TermInput } from './TermInput'

export type DialogMode = { kind: 'add' } | { kind: 'edit'; disease: UserDisease } | null

export interface SavedDisease {
  input: DiseaseInput
  labels?: Record<string, string>
}

interface Props {
  mode: DialogMode
  api: ApiHealth
  /** The user's other added diseases, so previews can match them too. */
  others: UserDisease[]
  onAdd: (items: SavedDisease[]) => void
  onUpdate: (id: string, item: SavedDisease) => void
  onClose: () => void
}

export function AddDiseaseDialog(props: Props) {
  const { mode, onClose } = props
  const dialog = useRef<HTMLDialogElement>(null)
  const [tab, setTab] = useState<'form' | 'json'>('form')
  const titleId = useId()

  useEffect(() => {
    const el = dialog.current
    if (!el) return
    if (mode && !el.open) el.showModal()
    if (!mode && el.open) el.close()
  }, [mode])

  return (
    <dialog ref={dialog} className="dialog" aria-labelledby={titleId} onClose={onClose}>
      {mode && (
        <>
          <header className="dialog-header">
            <h2 id={titleId}>{mode.kind === 'edit' ? `Edit ${mode.disease.input.name}` : 'Add a disease'}</h2>
            <button type="button" className="close-button" aria-label="Close" onClick={onClose}>
              ×
            </button>
          </header>
          {mode.kind === 'add' && (
            <div className="segmented dialog-tabs" role="tablist">
              <button type="button" role="tab" aria-selected={tab === 'form'} onClick={() => setTab('form')}>
                Fill in a form
              </button>
              <button type="button" role="tab" aria-selected={tab === 'json'} onClick={() => setTab('json')}>
                Upload JSON
              </button>
            </div>
          )}
          {mode.kind === 'edit' || tab === 'form' ? (
            <DiseaseForm key={mode.kind === 'edit' ? mode.disease.id : 'new'} {...props} initial={mode.kind === 'edit' ? mode.disease : undefined} />
          ) : (
            <JsonImport {...props} />
          )}
        </>
      )}
    </dialog>
  )
}

const apiBlocker = (api: ApiHealth) =>
  api.status === 'ready'
    ? null
    : api.status === 'loading'
      ? 'The v2 model is still loading.'
      : api.status === 'error'
        ? `The placement service could not load the v2 model: ${api.error}`
        : 'The placement service is not running, so matches cannot be computed. You can still save the disease and place it later.'

// --- Form ------------------------------------------------------------------------

const hpo = (token: string) => normaliseHpo(token) ?? { error: 'not an HPO ID; pick a suggestion or type e.g. HP:0001250' }
const ontology = (token: string) => {
  const t = token.trim()
  if (/^\d+$/.test(t)) return `ORPHA:${Number(t)}`
  return /^[A-Za-z]+[:_]\w+$/.test(t) ? t.replace('_', ':') : { error: 'use an ORPHA or MONDO ID, or pick a suggestion' }
}

function DiseaseForm({ api, others, onAdd, onUpdate, onClose, initial }: Props & { initial?: UserDisease }) {
  const start = initial?.input
  const [name, setName] = useState(start?.name ?? '')
  const [description, setDescription] = useState(start?.description ?? '')
  const [synonyms, setSynonyms] = useState<string[]>(start?.synonyms ?? [])
  const [phenotypes, setPhenotypes] = useState<Record<string, number>>(start ? phenotypeWeights(start) : {})
  const [genes, setGenes] = useState<string[]>(start?.genes ?? [])
  const [drugs, setDrugs] = useState<string[]>(start?.drugs ?? [])
  const [inheritance, setInheritance] = useState<string[]>(start?.inheritance ?? [])
  const [onset, setOnset] = useState<string[]>(start?.onset ?? [])
  const [parents, setParents] = useState<string[]>(start?.ontology_parents ?? [])
  const [prevalence, setPrevalence] = useState<string>(start?.prevalence ? String(Math.round(1 / start.prevalence)) : '')
  const [labels, setLabels] = useState<Record<string, string>>(initial?.labels ?? {})
  const [submitted, setSubmitted] = useState(false)
  const [preview, setPreview] = useState<{ result?: Placement; error?: string; busy?: boolean }>({})
  const onLabel = (id: string, label: string) => setLabels((l) => ({ ...l, [id]: label }))

  const input = useMemo<DiseaseInput>(() => {
    const n = Number(prevalence)
    return compactInput({
      name: name.trim(),
      description,
      synonyms,
      phenotypes,
      genes,
      drugs,
      inheritance,
      onset,
      ontology_parents: parents,
      prevalence: prevalence && n >= 1 ? 1 / n : undefined,
    })
  }, [name, description, synonyms, phenotypes, genes, drugs, inheritance, onset, parents, prevalence])

  const blocker = apiBlocker(api)
  const runPreview = async () => {
    if (!name.trim()) {
      setSubmitted(true)
      return
    }
    setPreview({ busy: true })
    try {
      const result = await placeDisease(
        initial?.id ?? 'USER:preview',
        input,
        others.filter((o) => o.id !== initial?.id).map((o) => ({ id: o.id, disease: o.input })),
        5,
      )
      setPreview({ result })
    } catch (e) {
      setPreview({ error: (e as Error).message })
    }
  }

  const save = () => {
    setSubmitted(true)
    if (!name.trim()) return
    const used = new Set([...Object.keys(phenotypes), ...drugs, ...parents])
    const keptLabels = Object.fromEntries(Object.entries(labels).filter(([k]) => used.has(k)))
    if (initial) onUpdate(initial.id, { input, labels: keptLabels })
    else onAdd([{ input, labels: keptLabels }])
    onClose()
  }

  const check = (list: string[], set: (v: string[]) => void, value: string) =>
    set(list.includes(value) ? list.filter((v) => v !== value) : [...list, value])

  return (
    <form
      className="dialog-body form-layout"
      onSubmit={(e) => {
        e.preventDefault()
        save()
      }}
    >
      <div className="form-fields">
        <label className="form-field">
          <span className="form-label">
            Name <span className="required">required</span>
          </span>
          <input value={name} onChange={(e) => setName(e.target.value)} aria-invalid={submitted && !name.trim()} autoFocus />
          {submitted && !name.trim() && <small className="warning-text">Give the disease a name.</small>}
        </label>

        <label className="form-field">
          <span className="form-label">
            Clinical description <span className="optional">optional</span>
          </span>
          <textarea rows={3} value={description} onChange={(e) => setDescription(e.target.value)} />
          <small className="muted">Free text. v2 compares it with Orphanet's clinical descriptions.</small>
        </label>

        <TermInput
          label="Phenotypes (HPO)"
          hint="Search by name or paste HPO IDs. Set how often each occurs; v2 weights phenotypes by it."
          placeholder="Seizure, HP:0001263…"
          field="phenotypes"
          normalise={hpo}
          values={Object.keys(phenotypes)}
          onChange={(v) => setPhenotypes(Object.fromEntries(v.map((t) => [t, phenotypes[t] ?? 0.5])))}
          weights={phenotypes}
          onWeight={(t, w) => setPhenotypes((p) => ({ ...p, [t]: w }))}
          labels={labels}
          onLabel={onLabel}
        />
        <TermInput
          label="Genes"
          hint="HGNC symbols, e.g. CDKL5."
          placeholder="CDKL5…"
          field="genes"
          values={genes}
          onChange={setGenes}
          labels={labels}
          onLabel={onLabel}
        />

        <div className="form-row two">
          <CheckGroup label="Inheritance" options={INHERITANCE_OPTIONS} values={inheritance} onChange={(v) => check(inheritance, setInheritance, v)} />
          <CheckGroup label="Age of onset" options={ONSET_OPTIONS} values={onset} onChange={(v) => check(onset, setOnset, v)} />
        </div>

        <div className="form-row two">
          <label className="form-field">
            <span className="form-label">
              Prevalence <span className="optional">optional</span>
            </span>
            <select
              value={PREVALENCE_CLASSES.find((c) => prevalence && Math.abs(1 / Number(prevalence) - c.value) / c.value < 0.01)?.value ?? ''}
              onChange={(e) => setPrevalence(e.target.value ? String(Math.round(1 / Number(e.target.value))) : '')}
            >
              <option value="">Unknown or exact below</option>
              {PREVALENCE_CLASSES.map((c) => (
                <option key={c.label} value={c.value}>
                  {c.label}
                </option>
              ))}
            </select>
          </label>
          <label className="form-field">
            <span className="form-label">
              Or exact: 1 in… <span className="optional">optional</span>
            </span>
            <input
              type="number"
              min={1}
              step={1}
              inputMode="numeric"
              placeholder="100000"
              value={prevalence}
              onChange={(e) => setPrevalence(e.target.value)}
            />
          </label>
        </div>

        <TermInput
          label="Drugs"
          hint="Drugs used or trialled: ChEMBL IDs or names. v2 matches names to ChEMBL."
          placeholder="Ganaxolone, CHEMBL…"
          field="drugs"
          values={drugs}
          onChange={setDrugs}
          labels={labels}
          onLabel={onLabel}
        />
        <TermInput
          label="Classification parents"
          hint="Orphanet or Mondo groups this disease belongs to, e.g. ORPHA:102369."
          placeholder="ORPHA:…"
          field="ontology"
          normalise={ontology}
          values={parents}
          onChange={setParents}
          labels={labels}
          onLabel={onLabel}
        />
        <TermInput
          label="Synonyms"
          hint="Other names. v2 compares name terms."
          placeholder="Other names…"
          values={synonyms}
          onChange={setSynonyms}
          labels={labels}
          onLabel={onLabel}
        />
      </div>

      <aside className="form-preview" aria-live="polite">
        <h3>Closest matches</h3>
        {blocker ? (
          <p className="muted small">{blocker}</p>
        ) : (
          <>
            <p className="muted small">Every field is optional; v2 uses whichever modalities you give.</p>
            <button type="button" className="button full" disabled={preview.busy || countFeatures(input) === 0} onClick={runPreview}>
              {preview.busy ? 'Finding…' : 'Preview matches'}
            </button>
          </>
        )}
        {preview.error && <p className="warning-text small">{preview.error}</p>}
        {preview.result && (
          <>
            <ol className="preview-list">
              {preview.result.neighbours.map((n) => (
                <li key={n.id}>
                  <span className={`line-key support-${n.support}`} aria-hidden />
                  <span>
                    {displayName(n.name)}
                    <span className="muted small">
                      {' '}
                      · {percentile(n.percentile)} · {modalityLabel(n.explanation[0]?.modality)}
                    </span>
                  </span>
                </li>
              ))}
            </ol>
            {preview.result.warnings.length > 0 && (
              <ul className="warnings small">
                {preview.result.warnings.map((w) => (
                  <li key={w}>{w}</li>
                ))}
              </ul>
            )}
          </>
        )}
      </aside>

      <footer className="dialog-footer">
        <button type="button" className="button" onClick={onClose}>
          Cancel
        </button>
        <button type="submit" className="button primary">
          {initial ? 'Save and place again' : blocker ? 'Save (place later)' : 'Add to graph'}
        </button>
      </footer>
    </form>
  )
}

function CheckGroup({ label, options, values, onChange }: { label: string; options: string[]; values: string[]; onChange: (value: string) => void }) {
  return (
    <fieldset className="form-field">
      <legend className="form-label">
        {label} <span className="optional">optional</span>
      </legend>
      <div className="check-grid">
        {options.map((o) => (
          <label key={o} className="check">
            <input type="checkbox" checked={values.includes(o)} onChange={() => onChange(o)} />
            {o}
          </label>
        ))}
      </div>
    </fieldset>
  )
}

// --- JSON upload -----------------------------------------------------------------

function JsonImport({ onAdd, onClose }: Props) {
  const [text, setText] = useState('')
  const [fileName, setFileName] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const inputId = useId()

  const parsed = useMemo(() => (text.trim() ? parseImport(text) : null), [text])
  const valid = (parsed?.items ?? []).filter((i): i is ImportItem & { input: DiseaseInput } => i.input !== null)

  const readFile = async (file: File) => {
    setFileName(file.name)
    setText(await file.text())
  }

  return (
    <div className="dialog-body">
      <p className="muted small">
        Upload a disease in v2's format, the same JSON <code>v2/place_disease.py --json</code> takes. One object, a
        list, or <code>{'{"diseases": [...]}'}</code>. Only <code>name</code> is required; the others are{' '}
        <code>description</code>, <code>synonyms</code>, <code>phenotypes</code> (list, or HPO term → share of
        patients), <code>genes</code>, <code>drugs</code>, <code>inheritance</code>, <code>onset</code>,{' '}
        <code>prevalence</code> (a fraction) and <code>ontology_parents</code>.{' '}
        <button type="button" className="link-button" onClick={() => download('disease-template.json', JSON.stringify(JSON_TEMPLATE, null, 2), 'application/json')}>
          Download a template
        </button>
      </p>

      <label
        htmlFor={inputId}
        className={`drop-zone${dragging ? ' is-dragging' : ''}`}
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragging(false)
          const file = e.dataTransfer.files[0]
          if (file) void readFile(file)
        }}
      >
        <strong>{fileName ?? 'Choose a .json file'}</strong>
        <span className="muted small">or drop it here</span>
        <input
          id={inputId}
          type="file"
          accept=".json,application/json"
          className="visually-hidden"
          onChange={(e) => {
            const file = e.target.files?.[0]
            if (file) void readFile(file)
          }}
        />
      </label>

      <label className="form-field">
        <span className="form-label">Or paste JSON</span>
        <textarea
          rows={6}
          className="mono"
          value={text}
          spellCheck={false}
          onChange={(e) => {
            setFileName(null)
            setText(e.target.value)
          }}
        />
      </label>

      {parsed?.error && <p className="callout">{parsed.error}</p>}
      {parsed && parsed.items.length > 0 && (
        <ul className="import-preview">
          {parsed.items.map((item, i) => (
            <li key={i}>
              <div className="import-row">
                <strong>{item.input?.name ?? `Item ${i + 1}`}</strong>
                {item.input && <span className="muted small">{plural(countFeatures(item.input), 'feature')}</span>}
              </div>
              {item.errors.map((e) => (
                <p key={e} className="warning-text small">
                  {e}
                </p>
              ))}
              {item.warnings.map((w) => (
                <p key={w} className="muted small">
                  {w}
                </p>
              ))}
            </li>
          ))}
        </ul>
      )}
      <p className="muted small">v2 checks every value when it places the disease and reports anything it ignores.</p>

      <footer className="dialog-footer">
        <button type="button" className="button" onClick={onClose}>
          Cancel
        </button>
        <button
          type="button"
          className="button primary"
          disabled={valid.length === 0}
          onClick={() => {
            onAdd(valid.map((i) => ({ input: i.input })))
            onClose()
          }}
        >
          {valid.length > 1 ? `Add ${valid.length} diseases` : 'Add to graph'}
        </button>
      </footer>
    </div>
  )
}
