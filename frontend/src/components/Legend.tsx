export function Legend() {
  return (
    <div className="legend">
      <span className="section-label">Reading the graph</span>
      <ul>
        <li>
          <span className="line-key support-curated" aria-hidden /> Curated relation backs the edge
        </li>
        <li>
          <span className="line-key support-plausible" aria-hidden /> Plausible: shared group, gene or drug
        </li>
        <li>
          <span className="line-key support-novel" aria-hidden /> Novel: hypothesis to review
        </li>
        <li>
          <span className="line-key support-novel is-user" aria-hidden /> Dotted: from a disease you added
        </li>
        <li>
          <span className="node-key filled" aria-hidden /> Orphanet disease
        </li>
        <li>
          <span className="node-key diamond" aria-hidden /> Added disease
        </li>
        <li>
          <span className="node-key sized" aria-hidden /> Larger node = more edges
        </li>
      </ul>
      <p className="muted small">
        Each disease links to its 10 most similar diseases under the v2 model. Nearby diseases in the layout have similar
        fused profiles.
      </p>
    </div>
  )
}
