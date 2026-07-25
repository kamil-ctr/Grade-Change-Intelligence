import { useState, useEffect } from 'react'
import { api } from '../lib/api'

const FIELDS = [
  { key: 'net_margin_per_tonne', label: 'Net margin ($/tonne)' },
  { key: 'rework_cost_per_tonne', label: 'Rework cost ($/tonne)' },
  { key: 'steam_cost_per_gj', label: 'Steam cost ($/GJ)' },
  { key: 'steam_gj_per_tonne', label: 'Steam intensity (GJ/tonne)' },
  { key: 'grade_changes_per_day', label: 'Grade changes / day' },
  { key: 'operating_days_per_year', label: 'Operating days / year' },
]

export default function EconomicsPanel({ economics, onSaved }) {
  const [values, setValues] = useState(economics || {})
  const [saved, setSaved] = useState(false)

  // `economics` arrives asynchronously (fetched in the parent), after this
  // component's first render -- useState's initial value only applies once,
  // so without this effect the form stays permanently empty.
  useEffect(() => {
    if (economics) setValues(economics)
  }, [economics])

  if (!economics) return null

  function set(key, v) {
    setSaved(false)
    setValues((s) => ({ ...s, [key]: v }))
  }

  async function save() {
    const updated = await api.updateEconomics(values)
    setValues(updated)
    setSaved(true)
    onSaved?.(updated)
  }

  return (
    <div>
      <div className="econ-form">
        {FIELDS.map((f) => (
          <div className="field" key={f.key}>
            <label>{f.label}</label>
            <input
              type="number"
              value={values[f.key] ?? ''}
              onChange={(e) => set(f.key, parseFloat(e.target.value))}
            />
          </div>
        ))}
      </div>
      <div className="econ-actions">
        <button className="btn accept" onClick={save}>Save assumptions</button>
        {saved && <span className="save-note">Applied to future pricing</span>}
      </div>
    </div>
  )
}
