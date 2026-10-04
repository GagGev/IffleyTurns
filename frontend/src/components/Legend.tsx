export function Legend() {
  return (
    <div className="legend">
      <span className="section-label">Reading the graph</span>
      <ul>
        <li>
          <span className="line-key rel-similar" aria-hidden /> Similar
        </li>
        <li>
          <span className="line-key rel-related-but-distinct" aria-hidden /> Related but distinct
        </li>
        <li>
          <span className="line-key rel-unrelated" aria-hidden /> Unrelated (control)
        </li>
        <li>
          <span className="line-key rel-computed" aria-hidden /> Computed similarity (added diseases)
        </li>
        <li>
          <span className="legend-width" aria-hidden /> Thicker line = more papers
        </li>
        <li>
          <span className="legend-fade" aria-hidden /> Stronger colour = higher score
        </li>
        <li>
          <span className="node-key filled" aria-hidden /> Rare disease
        </li>
        <li>
          <span className="node-key hollow" aria-hidden /> Common comparator
        </li>
        <li>
          <span className="node-key diamond" aria-hidden /> Disease you added
        </li>
        <li>
          <span className="node-key sized" aria-hidden /> Larger node = more pairs
        </li>
      </ul>
    </div>
  )
}
