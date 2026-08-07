import { useEffect, useState, useCallback } from 'react'
import { api } from './lib/api'
import ForecastCone from './components/ForecastCone'
import RecommendationsPanel from './components/RecommendationsPanel'
import CorrelationTable from './components/CorrelationTable'
import StabilizationBars from './components/StabilizationBars'
import TrustPanel from './components/TrustPanel'
import EconomicsPanel from './components/EconomicsPanel'

export default function App() {
  const [health, setHealth] = useState(null)
  const [healthError, setHealthError] = useState(null)
  const [healthRetryCount, setHealthRetryCount] = useState(0)
  const [events, setEvents] = useState([])
  const [eventId, setEventId] = useState(null)
  const [tMin, setTMin] = useState(12)
  const [live, setLive] = useState(null)
  // null = not yet fetched (loading state); [] = fetched, genuinely empty.
  // Panels need to tell those apart to show a skeleton vs. an empty-state
  // message instead of flashing "no data" before the first response lands.
  const [advisories, setAdvisories] = useState(null)
  const [correlations, setCorrelations] = useState(null)
  const [correlationsWarming, setCorrelationsWarming] = useState(false)
  const [stabilization, setStabilization] = useState(null)
  const [trust, setTrust] = useState(null)
  const [economics, setEconomics] = useState(null)
  const [error, setError] = useState(null)

  // /api/health fires alone, first. On a cold instance the other 7 calls
  // would otherwise stack up and all stall together; firing health first
  // and gating everything else on it turns 8 simultaneous cold-start
  // timeouts into one clear "waking up" signal, and by the time health
  // resolves the container is warm for what follows.
  useEffect(() => {
    let cancelled = false
    setHealthError(null)
    api.health().then((h) => { if (!cancelled) setHealth(h) })
      .catch((e) => { if (!cancelled) setHealthError(e.message) })
    return () => { cancelled = true }
  }, [healthRetryCount])

  useEffect(() => {
    if (!health) return
    api.events().then((evs) => {
      setEvents(evs)
      const riskiest = [...evs].sort((a, b) => b.max_abs_dev_pct - a.max_abs_dev_pct)[0]
      setEventId(riskiest?.event_id ?? evs[0]?.event_id)
    }).catch((e) => setError(e.message))
    api.correlations().then((r) => {
      setCorrelations(r.results)
      setCorrelationsWarming(r.warming)
    }).catch(() => {})
    api.economics().then(setEconomics).catch(() => {})
  }, [health])

  const refreshTrust = useCallback(() => {
    api.trust().then(setTrust).catch(() => {})
  }, [])

  useEffect(() => { if (health) refreshTrust() }, [health, refreshTrust])

  useEffect(() => {
    if (eventId == null) return
    api.live(eventId, tMin).then(setLive).catch((e) => setError(e.message))
    api.recommendations(eventId, tMin).then(setAdvisories).catch((e) => setError(e.message))
    api.stabilization(eventId).then(setStabilization).catch(() => {})
  }, [eventId, tMin])

  const current = events.find((e) => e.event_id === eventId)

  // Paints immediately on mount (health starts null, no API call awaited)
  // and stays up until /api/health resolves -- the one moment in the whole
  // app where the user would otherwise see a blank page.
  if (!health) {
    return (
      <div className="app">
        <div className="connecting-screen">
          <div className="brand" style={{ flexDirection: 'column', gap: 4 }}>
            <h1>Grade Change Intelligence</h1>
            <span className="tagline">Advisory only &middot; never writes to the control system</span>
          </div>
          <div className="connecting-card">
            <div className="spinner" />
            <p className="connecting-message">
              {healthError
                ? `Server didn't respond after several tries (${healthError}).`
                : 'Waking up the server — first load can take up to a minute.'}
            </p>
            {healthError ? (
              <button className="btn" onClick={() => setHealthRetryCount((c) => c + 1)}>
                Retry
              </button>
            ) : (
              <>
                <div className="skeleton-line long" />
                <div className="skeleton-line medium" />
                <div className="skeleton-line short" />
              </>
            )}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="app">
      <div className="header">
        <div className="brand">
          <h1>Grade Change Intelligence</h1>
          <span className="tagline">Advisory only &middot; never writes to the control system</span>
        </div>
        <div className="controls">
          {health && (
            <span className={`mode-pill ${health.risk_model_loaded && health.forecast_model_loaded ? 'ok' : 'warn'}`}>
              {health.risk_model_loaded && health.forecast_model_loaded ? 'models live' : 'degraded mode'}
            </span>
          )}
          {current && (
            <span className="grade-badge">
              {current.from_grade} <span className="arrow">&rarr;</span> {current.to_grade}
            </span>
          )}
          <div className="field">
            <label>Transition</label>
            <select value={eventId ?? ''} onChange={(e) => setEventId(Number(e.target.value))}>
              {events.map((e) => (
                <option key={e.event_id} value={e.event_id}>
                  #{e.event_id}: {e.from_grade}&rarr;{e.to_grade} {e.off_spec ? '(off-spec)' : ''}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>Clock: t = {tMin.toFixed(2)} min</label>
            <input type="range" min="0" max="29" step="0.5" value={tMin} onChange={(e) => setTMin(Number(e.target.value))} />
          </div>
        </div>
      </div>

      {error && <div className="empty-state" style={{ color: 'var(--status-critical)' }}>{error}</div>}

      <div className="grid">
        <div className="panel span-4">
          <div className="panel-head">
            <div>
              <h2>Basis-Weight Off-Spec Risk</h2>
              <div className="subtitle">Current deviation and breach probability</div>
            </div>
            <span className="deliverable-tag">Deliverable 3</span>
          </div>
          {live ? (
            <div className="stat-row">
              <div className="stat-tile">
                <div className={`value ${Math.abs(live.basis_weight_dev_pct) > 2.5 ? 'critical' : 'good'}`}>
                  {live.basis_weight_dev_pct.toFixed(1)}%
                </div>
                <div className="label">Current deviation</div>
              </div>
              {live.risk_probability !== undefined && (
                <div className="stat-tile">
                  <span className={`risk-alarm ${live.risk_alarm ? 'critical' : 'good'}`}>
                    {live.risk_alarm ? 'BREACH RISK' : 'in spec'} &middot; {(live.risk_probability * 100).toFixed(1)}%
                  </span>
                  <div className="label" style={{ marginTop: 6 }}>P(breach within 10 min)</div>
                </div>
              )}
            </div>
          ) : (
            <div>
              <div className="skeleton-line long" />
              <div className="skeleton-line medium" />
            </div>
          )}
        </div>

        <div className="panel span-4">
          <div className="panel-head">
            <div>
              <h2>Basis-Weight Forecast (+2 / +5 / +10 min)</h2>
              <div className="subtitle">Quantile forecast cone</div>
            </div>
            <span className="deliverable-tag">Deliverable 3</span>
          </div>
          <ForecastCone live={live} />
        </div>

        <div className="panel span-4">
          <div className="panel-head">
            <div>
              <h2>Recommendations</h2>
              <div className="subtitle">Priced, provenanced, accept/reject</div>
            </div>
            <span className="deliverable-tag">3 &middot; 5 &middot; 6</span>
          </div>
          <RecommendationsPanel advisories={advisories} onDecided={refreshTrust} />
        </div>

        <div className="panel span-6">
          <div className="panel-head">
            <div>
              <h2>Correlation explorer</h2>
              <div className="subtitle">Lagged relationships discovered from the historical corpus</div>
            </div>
            <span className="deliverable-tag">Deliverable 3</span>
          </div>
          <CorrelationTable correlations={correlations} warming={correlationsWarming} />
        </div>

        <div className="panel span-6">
          <div className="panel-head">
            <div>
              <h2>Stabilization impact</h2>
              <div className="subtitle">Which plan parameters cut settling time fastest, this transition</div>
            </div>
            <span className="deliverable-tag">Deliverable 4</span>
          </div>
          <StabilizationBars impacts={stabilization} />
        </div>

        <div className="panel span-6">
          <div className="panel-head">
            <div>
              <h2>Trust &amp; ledger</h2>
              <div className="subtitle">Acceptance rate and confidence calibration</div>
            </div>
            <span className="deliverable-tag">Deliverable 6</span>
          </div>
          <TrustPanel trust={trust} />
        </div>

        <div className="panel span-6">
          <div className="panel-head">
            <div>
              <h2>ROI assumptions</h2>
              <div className="subtitle">Editable economics feeding every dollar figure above</div>
            </div>
            <span className="deliverable-tag">ROI engine</span>
          </div>
          <EconomicsPanel economics={economics} onSaved={setEconomics} />
        </div>
      </div>

      <footer className="policy-note">
        ADVISORY mode &middot; no control writeback &middot; maturity ladder: ADVISORY &rarr; SUPERVISORY &rarr; CLOSED_LOOP
      </footer>
    </div>
  )
}
