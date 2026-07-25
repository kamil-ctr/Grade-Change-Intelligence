export default function CorrelationTable({ correlations }) {
  if (!correlations || correlations.length === 0) {
    return <div className="empty-state">No correlations above threshold in this sample.</div>
  }
  const top = correlations.slice(0, 10)
  const maxAbs = Math.max(...top.map((c) => Math.abs(c.correlation)), 0.01)

  return (
    <table className="data-table">
      <thead>
        <tr>
          <th>Relationship</th>
          <th>Lag</th>
          <th>Strength</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        {top.map((c) => (
          <tr key={`${c.cause}-${c.effect}`}>
            <td>
              <strong>{c.cause}</strong> <span style={{ color: 'var(--text-muted)' }}>&rarr;</span> {c.effect}
            </td>
            <td style={{ color: 'var(--text-secondary)' }}>{c.best_lag_min.toFixed(2)} min</td>
            <td>
              <div className="bar-track" style={{ width: 80 }}>
                <div
                  className="bar-fill"
                  style={{
                    width: `${(Math.abs(c.correlation) / maxAbs) * 100}%`,
                    background: c.correlation >= 0 ? 'var(--series-1)' : 'var(--series-8)',
                  }}
                />
              </div>
            </td>
            <td>
              {c.novel ? <span className="novel-tag">NOVEL</span> : <span className="known-tag">known loop</span>}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
