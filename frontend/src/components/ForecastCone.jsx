const SPEC_PCT = 2.5
const WIDTH = 560
const HEIGHT = 230
const MARGIN = { top: 14, right: 18, bottom: 30, left: 38 }

function scaleX(xMin, xMax) {
  const w = WIDTH - MARGIN.left - MARGIN.right
  return (x) => MARGIN.left + ((x - xMin) / (xMax - xMin)) * w
}
function scaleY(yMin, yMax) {
  const h = HEIGHT - MARGIN.top - MARGIN.bottom
  return (y) => MARGIN.top + h - ((y - yMin) / (yMax - yMin)) * h
}

export default function ForecastCone({ live }) {
  if (!live) {
    return (
      <div>
        <div className="skeleton-line long" />
        <div className="skeleton-line medium" />
        <div className="skeleton-line short" />
      </div>
    )
  }
  const cone = live.forecast_cone
  const now = live.basis_weight_dev_pct ?? 0

  if (!cone) {
    return (
      <div className="empty-state">
        Forecast model not loaded on this run -- see <code>/api/health</code>.
      </div>
    )
  }

  const horizons = [
    { x: 0, p10: now, p50: now, p90: now },
    { x: 2, ...cone['2min'] && { p10: cone['2min']['0.1'], p50: cone['2min']['0.5'], p90: cone['2min']['0.9'] } },
    { x: 5, ...cone['5min'] && { p10: cone['5min']['0.1'], p50: cone['5min']['0.5'], p90: cone['5min']['0.9'] } },
    { x: 10, ...cone['10min'] && { p10: cone['10min']['0.1'], p50: cone['10min']['0.5'], p90: cone['10min']['0.9'] } },
  ].filter((p) => p.p10 !== undefined)

  const allY = horizons.flatMap((p) => [p.p10, p.p90]).concat([SPEC_PCT, -SPEC_PCT, now])
  const yMax = Math.max(...allY) * 1.15 + 0.5
  const yMin = Math.min(...allY) * 1.15 - 0.5
  const xScale = scaleX(0, 10)
  const yScale = scaleY(yMin, yMax)

  const bandPath =
    `M ${horizons.map((p) => `${xScale(p.x)},${yScale(p.p90)}`).join(' L ')} ` +
    `L ${[...horizons].reverse().map((p) => `${xScale(p.x)},${yScale(p.p10)}`).join(' L ')} Z`
  const medianPath = `M ${horizons.map((p) => `${xScale(p.x)},${yScale(p.p50)}`).join(' L ')}`

  const yTicks = [yMin, yMin + (yMax - yMin) / 2, yMax].map((v) => Math.round(v * 10) / 10)
  const xTicks = [0, 2, 5, 10]

  return (
    <div>
      <svg className="chart-svg" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} width="100%" role="img"
        aria-label="Forecast cone of basis weight deviation over the next 10 minutes">
        {yTicks.map((v) => (
          <g key={v}>
            <line className="gridline" x1={MARGIN.left} x2={WIDTH - MARGIN.right} y1={yScale(v)} y2={yScale(v)} />
            <text x={MARGIN.left - 8} y={yScale(v) + 3} textAnchor="end">{v}%</text>
          </g>
        ))}
        <line className="spec-line" x1={MARGIN.left} x2={WIDTH - MARGIN.right} y1={yScale(SPEC_PCT)} y2={yScale(SPEC_PCT)} />
        <line className="spec-line" x1={MARGIN.left} x2={WIDTH - MARGIN.right} y1={yScale(-SPEC_PCT)} y2={yScale(-SPEC_PCT)} />

        <path className="band" d={bandPath} />
        <path className="median-line" d={medianPath} />
        <circle className="dot" cx={xScale(0)} cy={yScale(now)} r="4" />

        <line className="axis-line" x1={MARGIN.left} x2={WIDTH - MARGIN.right} y1={HEIGHT - MARGIN.bottom} y2={HEIGHT - MARGIN.bottom} />
        {xTicks.map((t) => (
          <text key={t} x={xScale(t)} y={HEIGHT - MARGIN.bottom + 16} textAnchor="middle">
            {t === 0 ? 'now' : `+${t}m`}
          </text>
        ))}
      </svg>
      <div className="legend">
        <span className="legend-item"><span className="swatch" style={{ background: 'var(--series-1)' }} />Current deviation</span>
        <span className="legend-item"><span className="swatch" style={{ background: 'var(--series-2)' }} />Median forecast</span>
        <span className="legend-item"><span className="swatch band" style={{ background: 'var(--series-2)' }} />P10-P90 range</span>
        <span className="legend-item"><span className="swatch" style={{ background: 'var(--status-critical)', opacity: 0.6 }} />+/-2.5% spec</span>
      </div>
    </div>
  )
}
