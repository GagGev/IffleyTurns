import { useDeferredValue, useEffect, useId, useMemo, useRef, useState } from 'react'
import { FAMILY_COLUMNS, type FeatureCatalogue, type FeatureFamily, type FeatureProfile } from '../data/features'
import { plural } from '../lib/format'
import { featureCount, rankNeighbours, type Weights } from '../lib/similarity'
import {
  emptyUserDisease,
  type FamilyColumn,
  type ImportItem,
  INHERITANCE_OPTIONS,
  JSON_TEMPLATE,
  normaliseOrpha,
  ONSET_OPTIONS,
  parseImport,
  PREVALENCE_CLASSES,
  toProfile,
  type UserDisease,
  type UserStatus,
} from '../lib/userDiseases'
import { NeighbourList } from './NeighbourList'
import { TermInput } from './TermInput'

export type DialogMode = { kind: 'add' } | { kind: 'edit'; disease: UserDisease } | null

interface Props {
  mode: DialogMode
  catalogue: FeatureCatalogue | null
  /** Profiles to preview matches against. */
  candidates: FeatureProfile[]
  weights: Weights
  onAdd: (items: UserDisease[]) => void
  onUpdate: (disease: UserDisease) => void
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
            <h2 id={titleId}>{mode.kind === 'edit' ? `Edit ${mode.disease.name}` : 'Add a disease'}</h2>
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
            <DiseaseForm
              key={mode.kind === 'edit' ? mode.disease.id : 'new'}
              {...props}
              initial={mode.kind === 'edit' ? mode.disease : undefined}
            />
          ) : (
            <JsonImport {...props} />
          )}
        </>
      )}
    </dialog>
  )
}

// --- Form ------------------------------------------------------------------------

const TERM_FIELDS: { family: FeatureFamily; label: string; hint: string; placeholder: string }[] = [
  {
    family: 'phenotypes',
    label: 'Phenotypes (HPO)',
    hint: 'HPO IDs such as HP:0001166, or names when feature profiles are loaded. Paste a list to add many.',
    placeholder: 'HP:0001166, HP:0001519…',
  },
  { family: 'genes', label: 'Associated genes', hint: 'HGNC symbols, e.g. FBN1.', placeholder: 'FBN1, TGFBR2…' },
  {
    family: 'body_systems',
    label: 'Body systems',
    hint: 'Orphanet classification heads, e.g. ORPHA:98006, or pick a suggestion.',
    placeholder: 'ORPHA:…',
  },
  {
    family: 'classifications',
    label: 'Disease categories',
    hint: 'Orphanet or ontology parent IDs.',
    placeholder: 'ORPHA:…',
  },
  {
    family: 'approved_drugs',
    label: 'Approved drugs',
    hint: 'ChEMBL IDs (e.g. CHEMBL1201580), or drug names when feature profiles are loaded.',
    placeholder: 'CHEMBL…',
  },
]

function DiseaseForm({ catalogue, candidates, weights, onAdd, onUpdate, onClose, initial }: Props & { initial?: UserDisease }) {
  const [draft, setDraft] = useState<UserDisease>(() => initial ?? emptyUserDisease())
  const [exactPrevalence, setExactPrevalence] = useState(() =>
    initial?.prevalence_estimated_per_person ? String(Math.round(1 / initial.prevalence_estimated_per_person)) : '',
  )
  const [submitted, setSubmitted] = useState(false)
  const set = <K extends keyof UserDisease>(key: K, value: UserDisease[K]) => setDraft((d) => ({ ...d, [key]: value }))
  const formId = useId()

  const inheritanceTerms = useMemo(() => catalogueTerms(catalogue, 'inheritance', INHERITANCE_OPTIONS), [catalogue])
  const onsetTerms = useMemo(() => catalogueTerms(catalogue, 'onset', ONSET_OPTIONS), [catalogue])
  // Keep values that came from an upload even when they are not in the list.
  const withValues = (list: string[], values: string[]) => [...list, ...values.filter((v) => !list.includes(v))]
  const inheritanceOptions = withValues(inheritanceTerms, draft.inheritance)
  const onsetOptions = withValues(onsetTerms, draft.onset)

  const orpha = draft.orpha_id ? normaliseOrpha(draft.orpha_id) : null
  const catalogueMatch = orpha ? catalogue?.byId.get(orpha) : undefined

  const fillFromCatalogue = () => {
    if (!catalogueMatch) return
    setDraft((d) => {
      const next = { ...d, name: d.name || catalogueMatch.name }
      for (const [family, column] of Object.entries(FAMILY_COLUMNS) as [FeatureFamily, FamilyColumn][]) {
        const existing = d[column] as string[]
        ;(next[column] as string[]) = [...new Set([...existing, ...(catalogueMatch.sets[family] ?? [])])]
      }
      if (!d.prevalence_class && catalogueMatch.prevalenceClass) next.prevalence_class = catalogueMatch.prevalenceClass
      return next
    })
  }

  const deferred = useDeferredValue(draft)
  const preview = useMemo(() => {
    const profile = toProfile({ ...deferred, id: deferred.id || 'user:draft', name: deferred.name || 'This disease' }, catalogue)
    if (featureCount(profile) === 0) return null
    const others = candidates.filter((c) => c.id !== deferred.id)
    return { profile, neighbours: rankNeighbours(profile, others, 5, weights) }
  }, [deferred, catalogue, candidates, weights])

  const nameMissing = !draft.name.trim()
  const save = () => {
    setSubmitted(true)
    if (nameMissing) return
    const n = Number(exactPrevalence)
    const disease: UserDisease = {
      ...draft,
      name: draft.name.trim(),
      orpha_id: orpha ?? undefined,
      prevalence_estimated_per_person: exactPrevalence && n >= 1 ? 1 / n : undefined,
    }
    if (initial) onUpdate(disease)
    else onAdd([disease])
    onClose()
  }

  return (
    <form
      id={formId}
      className="dialog-body form-layout"
      onSubmit={(e) => {
        e.preventDefault()
        save()
      }}
    >
      <div className="form-fields">
        <div className="form-row two">
          <label className="form-field">
            <span className="form-label">
              Name <span className="required">required</span>
            </span>
            <input
              value={draft.name}
              onChange={(e) => set('name', e.target.value)}
              aria-invalid={submitted && nameMissing}
              autoFocus
            />
            {submitted && nameMissing && <small className="warning-text">Give the disease a name.</small>}
          </label>
          <label className="form-field">
            <span className="form-label">
              ORPHA ID <span className="optional">optional</span>
            </span>
            <input
              value={draft.orpha_id ?? ''}
              placeholder="ORPHA:558"
              onChange={(e) => set('orpha_id', e.target.value || undefined)}
            />
            {draft.orpha_id && !orpha && <small className="warning-text">Not an ORPHA ID.</small>}
            {catalogueMatch && (
              <button type="button" className="link-button small" onClick={fillFromCatalogue}>
                Copy features from {catalogueMatch.name}
              </button>
            )}
          </label>
        </div>

        <fieldset className="form-field">
          <legend className="form-label">Rarity</legend>
          <div className="radio-row">
            {(['rare', 'common', 'unknown'] as UserStatus[]).map((s) => (
              <label key={s} className="check">
                <input type="radio" name={`${formId}-status`} checked={draft.status === s} onChange={() => set('status', s)} />
                {s === 'unknown' ? 'Not sure' : s[0].toUpperCase() + s.slice(1)}
              </label>
            ))}
          </div>
        </fieldset>

        {TERM_FIELDS.slice(0, 2).map((f) => (
          <TermInput
            key={f.family}
            {...f}
            catalogue={catalogue}
            values={draft[FAMILY_COLUMNS[f.family] as FamilyColumn] as string[]}
            onChange={(v) => set(FAMILY_COLUMNS[f.family] as FamilyColumn, v)}
          />
        ))}

        <div className="form-row two">
          <CheckGroup
            label="Inheritance"
            options={inheritanceOptions}
            values={draft.inheritance}
            onChange={(v) => set('inheritance', v)}
          />
          <CheckGroup label="Age of onset" options={onsetOptions} values={draft.onset} onChange={(v) => set('onset', v)} />
        </div>

        <div className="form-row two">
          <label className="form-field">
            <span className="form-label">
              Prevalence class <span className="optional">optional</span>
            </span>
            <select value={draft.prevalence_class ?? ''} onChange={(e) => set('prevalence_class', e.target.value || undefined)}>
              <option value="">Unknown</option>
              {PREVALENCE_CLASSES.map((c) => (
                <option key={c} value={c}>
                  {c}
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
              placeholder="50000"
              value={exactPrevalence}
              onChange={(e) => setExactPrevalence(e.target.value)}
            />
            <small className="muted">Used instead of the class when given.</small>
          </label>
        </div>

        {TERM_FIELDS.slice(2).map((f) => (
          <TermInput
            key={f.family}
            {...f}
            catalogue={catalogue}
            values={draft[FAMILY_COLUMNS[f.family] as FamilyColumn] as string[]}
            onChange={(v) => set(FAMILY_COLUMNS[f.family] as FamilyColumn, v)}
          />
        ))}

        <label className="form-field">
          <span className="form-label">
            Notes <span className="optional">optional, not used for scoring</span>
          </span>
          <textarea rows={2} value={draft.notes ?? ''} onChange={(e) => set('notes', e.target.value || undefined)} />
        </label>
      </div>

      <aside className="form-preview" aria-live="polite">
        <h3>Closest matches</h3>
        {!preview ? (
          <p className="muted small">Add any feature to see which diseases it resembles. Every feature is optional; the score uses whichever ones you give.</p>
        ) : (
          <NeighbourList
            target={preview.profile}
            neighbours={preview.neighbours}
            weights={weights}
            empty={
              candidates.length === 0
                ? 'Nothing to compare with yet: feature profiles for catalogue diseases have not been generated, and no other diseases have been added.'
                : 'No disease shares any of these features yet.'
            }
          />
        )}
      </aside>

      <footer className="dialog-footer">
        <button type="button" className="button" onClick={onClose}>
          Cancel
        </button>
        <button type="submit" className="button primary">
          {initial ? 'Save changes' : 'Add to graph'}
        </button>
      </footer>
    </form>
  )
}

/** Every term the catalogue uses for a family, or the Orphanet list when there is no catalogue. */
function catalogueTerms(catalogue: FeatureCatalogue | null, family: FeatureFamily, fallback: string[]): string[] {
  const terms = new Set<string>()
  for (const p of catalogue?.profiles ?? []) for (const t of p.sets[family] ?? []) terms.add(t)
  return terms.size ? [...terms].sort() : fallback
}

function CheckGroup({
  label,
  options,
  values,
  onChange,
}: {
  label: string
  options: string[]
  values: string[]
  onChange: (values: string[]) => void
}) {
  return (
    <fieldset className="form-field">
      <legend className="form-label">
        {label} <span className="optional">optional</span>
      </legend>
      <div className="check-grid">
        {options.map((o) => (
          <label key={o} className="check">
            <input
              type="checkbox"
              checked={values.includes(o)}
              onChange={() => onChange(values.includes(o) ? values.filter((v) => v !== o) : [...values, o])}
            />
            {o}
          </label>
        ))}
      </div>
    </fieldset>
  )
}

// --- JSON upload -----------------------------------------------------------------

function JsonImport({ catalogue, onAdd, onClose }: Props) {
  const [text, setText] = useState('')
  const [fileName, setFileName] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const inputId = useId()

  const parsed = useMemo(() => (text.trim() ? parseImport(text, catalogue) : null), [text, catalogue])
  const valid = (parsed?.items ?? []).filter((i): i is ImportItem & { disease: UserDisease } => i.disease !== null)

  const readFile = async (file: File) => {
    setFileName(file.name)
    setText(await file.text())
  }

  const downloadTemplate = () => {
    const blob = new Blob([JSON.stringify(JSON_TEMPLATE, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'disease-template.json'
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="dialog-body">
      <p className="muted small">
        One disease object, a list of them, or <code>{'{"diseases": [...]}'}</code>. Field names follow{' '}
        <code>diseases.parquet</code>: <code>name</code> (required), <code>orpha_id</code>, <code>hpo_ids</code>,{' '}
        <code>gene_symbols</code>, <code>inheritance</code>, <code>onset</code>, <code>category_ids</code>,{' '}
        <code>body_system_ids</code>, <code>approved_drug_ids</code>, <code>prevalence_class</code> or{' '}
        <code>prevalence_estimated_per_person</code>. Everything except the name is optional.{' '}
        <button type="button" className="link-button" onClick={downloadTemplate}>
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
                <strong>{item.disease?.name ?? `Item ${i + 1}`}</strong>
                {item.disease && (
                  <span className="muted small">
                    {plural(featureCount(toProfile({ ...item.disease, id: 'user:preview' }, catalogue)), 'feature')}
                  </span>
                )}
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

      <footer className="dialog-footer">
        <button type="button" className="button" onClick={onClose}>
          Cancel
        </button>
        <button
          type="button"
          className="button primary"
          disabled={valid.length === 0}
          onClick={() => {
            onAdd(valid.map((i) => i.disease))
            onClose()
          }}
        >
          {valid.length > 1 ? `Add ${valid.length} diseases` : 'Add to graph'}
        </button>
      </footer>
    </div>
  )
}
