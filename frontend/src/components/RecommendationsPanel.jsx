import { useState } from 'react'
import { api } from '../lib/api'

const fmtUsd = (v) => `$${Math.round(v).toLocaleString()}`

export default function RecommendationsPanel({ advisories, onDecided }) {
  const [decisions, setDecisions] = useState({})
  const [pending, setPending] = useState(null)

  if (!advisories || advisories.length === 0) {
    return <div className="empty-state">Nothing meets the surfacing policy for this transition right now.</div>
  }

  async function decide(id, decision) {
    setPending(id)
    try {
      await api.feedback(id, decision)
      setDecisions((d) => ({ ...d, [id]: decision }))
      onDecided?.()
    } catch (e) {
      console.error(e)
    } finally {
      setPending(null)
    }
  }

  return (
    <div>
      {advisories.map((a) => {
        const decision = decisions[a.id]
        return (
          <div className="advisory-card" key={a.id}>
            <div className="top-row">
              <div className="title">{a.title}</div>
              <span className={`source-tag ${a.source}`}>{a.source_label}</span>
            </div>
            <div className="explanation">{a.explanation}</div>
            {a.value && (
              <div className="value-row">
                <span className="amount">{fmtUsd(a.value.point_estimate_usd)}</span>
                <span className="band">
                  P10 {fmtUsd(a.value.low_usd)} &ndash; P90 {fmtUsd(a.value.high_usd)}
                  {a.value.annualized_usd ? ` -- ${fmtUsd(a.value.annualized_usd)}/yr if typical` : ''}
                </span>
              </div>
            )}
            <div className="confidence">confidence {(a.confidence * 100).toFixed(0)}%</div>
            <div className="btn-row" style={{ marginTop: 8 }}>
              {decision ? (
                <span className={`decision-tag ${decision}`}>{decision === 'accepted' ? 'Accepted' : 'Rejected'}</span>
              ) : (
                <>
                  <button className="btn accept" disabled={pending === a.id} onClick={() => decide(a.id, 'accepted')}>
                    Accept
                  </button>
                  <button className="btn reject" disabled={pending === a.id} onClick={() => decide(a.id, 'rejected')}>
                    Reject
                  </button>
                </>
              )}
            </div>
          </div>
        )
      })}
    </div>
  )
}
