const PARAM_LABEL = {
  ramp_min: 'Ramp duration',
  lead_scale: 'Lead compensation',
  tau_c_scale: 'Trim aggressiveness',
}

export default function StabilizationBars({ impacts }) {
  if (!impacts || impacts.length === 0) {
    return <div className="empty-state">No stabilization data for this transition.</div>
  }
  const maxAbs = Math.max(...impacts.map((i) => Math.abs(i.sensitivity_min_per_unit)), 0.01)

  return (
    <div>
      {impacts.map((imp) => {
        const frac = Math.abs(imp.sensitivity_min_per_unit) / maxAbs
        const color = imp.best_direction === 'none' ? 'var(--text-muted)' : 'var(--series-1)'
        return (
          <div className="bar-row" key={imp.parameter}>
            <div>
              <div className="row-label">{PARAM_LABEL[imp.parameter] || imp.parameter}</div>
              <div className="row-sub">
                {imp.best_direction === 'none'
                  ? 'no improving direction found'
                  : `${imp.best_direction} ${Math.abs(imp.best_delta).toFixed(2)} -> -${imp.improvement_min.toFixed(2)} min settle`}
              </div>
            </div>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${frac * 100}%`, background: color }} />
            </div>
            <div className="row-value">{Math.abs(imp.sensitivity_min_per_unit).toFixed(2)} min/unit</div>
          </div>
        )
      })}
    </div>
  )
}
