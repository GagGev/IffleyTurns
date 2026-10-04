import type { EdgeDetail } from '../data/types'
import { modalityLabel, score } from '../lib/format'

interface Props {
  detail: EdgeDetail
  modalities: string[]
  descriptions: Record<string, string>
}

const RELATION_TITLES: Record<string, string> = {
  orphanet: 'Orphanet relation',
  gene: 'Shared causal gene',
  drug: 'Shared trial drug',
}

/** Why v2 scores a pair as similar: shared features, then each modality's contribution. */
export function EdgeExplanation({ detail, modalities, descriptions }: Props) {
  const relations = Object.entries(detail.relations).filter(([, v]) => v)
  const maxContribution = Math.max(...modalities.map((m) => detail.contributions[m] ?? 0), 1e-9)
  const rows = modalities
    .map((m) => ({ m, similarity: detail.similarities[m], contribution: detail.contributions[m] ?? 0 }))
    .sort((a, b) => b.contribution - a.contribution || (b.similarity ?? -1) - (a.similarity ?? -1))

  return (
    <>
      {relations.length > 0 && (
        <section className="panel-section">
          <h3>Curated relations</h3>
          <dl className="relations">
            {relations.map(([k, v]) => (
              <div key={k}>
                <dt>{RELATION_TITLES[k] ?? k}</dt>
                <dd>{v}</dd>
              </div>
            ))}
          </dl>
        </section>
      )}

      <section className="panel-section">
        <h3>What they share</h3>
        {detail.evidence.length === 0 ? (
          <p className="muted">No modality contributes positively; the score comes from the annotation pattern alone.</p>
        ) : (
          <ul className="evidence-list">
            {detail.evidence.map((e) => (
              <li key={e.modality}>
                <div className="evidence-head">
                  <strong title={descriptions[e.modality]}>{modalityLabel(e.modality)}</strong>
                  <span className="num muted">+{score(e.contribution)}</span>
                </div>
                {e.shared.length > 0 && (
                  <span className="shared-terms">
                    {e.shared.map((s) => (
                      <span key={s} className="chip" title={s}>
                        {s}
                      </span>
                    ))}
                  </span>
                )}
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="panel-section">
        <h3>Contribution by modality</h3>
        <table className="contributions">
          <caption className="visually-hidden">Similarity and score contribution for each of the model's modalities</caption>
          <thead>
            <tr>
              <th scope="col">Modality</th>
              <th scope="col" className="num">
                Similarity
              </th>
              <th scope="col">Contribution</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(({ m, similarity, contribution }) => (
              <tr key={m} className={similarity === null ? 'is-unavailable' : undefined}>
                <th scope="row" title={descriptions[m]}>
                  {modalityLabel(m)}
                </th>
                <td className="num">{similarity === null ? '–' : score(similarity)}</td>
                <td className="bar-cell">
                  {similarity === null ? (
                    <span className="no-evidence">Not annotated for both</span>
                  ) : contribution > 0 ? (
                    <span className="bar-row">
                      <span className="bar-track">
                        <span className="bar-fill" style={{ width: `${Math.max((contribution / maxContribution) * 100, 2)}%` }} />
                      </span>
                      <span className="num">+{score(contribution)}</span>
                    </span>
                  ) : (
                    <span className="no-evidence">0</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="muted small">
          The score is the sum of these contributions, an intercept, and an adjustment of{' '}
          {detail.annotationAdjustment >= 0 ? '+' : ''}
          {score(detail.annotationAdjustment)} for which modalities both diseases have. Contributions are non-negative, so
          each one is evidence for similarity.
        </p>
      </section>
    </>
  )
}
