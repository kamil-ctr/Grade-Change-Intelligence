const pct = (v) => (Number.isFinite(v) ? `${Math.round(v * 100)}%` : '--')
const fmtUsd = (v) => `$${Math.round(v).toLocaleString()}`

export default function TrustPanel({ trust }) {
  if (!trust) return null
  return (
    <div>
      <div className="stat-row">
        <div className="stat-tile">
          <div className="value">{trust.n_surfaced}</div>
          <div className="label">Advisories surfaced</div>
        </div>
        <div className="stat-tile">
          <div className="value">{trust.n_responded}</div>
          <div className="label">Responded</div>
        </div>
        <div className="stat-tile">
          <div className="value good">{pct(trust.acceptance_rate_overall)}</div>
          <div className="label">Acceptance rate</div>
        </div>
        <div className="stat-tile">
          <div className="value">{fmtUsd(trust.realized_value_usd_accepted || 0)}</div>
          <div className="label">Realized value (accepted)</div>
        </div>
      </div>

      <div style={{ marginTop: 16, fontSize: 12.5, color: 'var(--text-secondary)' }}>
        Mean confidence &mdash; accepted <strong>{pct(trust.mean_confidence_accepted)}</strong> vs rejected{' '}
        <strong>{pct(trust.mean_confidence_rejected)}</strong>
        {Number.isFinite(trust.mean_confidence_accepted) && Number.isFinite(trust.mean_confidence_rejected) && (
          <span style={{ color: 'var(--text-muted)' }}>
            {' '}
            {trust.mean_confidence_accepted > trust.mean_confidence_rejected
              ? '-- confidence tracks operator trust, as intended.'
              : '-- confidence is not yet separating accepted from rejected advice.'}
          </span>
        )}
      </div>

      {trust.acceptance_rate_by_source && Object.keys(trust.acceptance_rate_by_source).length > 0 && (
        <div style={{ marginTop: 14 }}>
          {Object.entries(trust.acceptance_rate_by_source).map(([source, rate]) => (
            <div className="bar-row" key={source}>
              <div className="row-label">{source.replace('_', ' ')}</div>
              <div className="bar-track">
                <div className="bar-fill" style={{ width: `${rate * 100}%`, background: 'var(--series-3)' }} />
              </div>
              <div className="row-value">{pct(rate)}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
