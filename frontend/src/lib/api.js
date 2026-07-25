const BASE = '/api'

async function request(path, options) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail || `${res.status} ${res.statusText}`)
  }
  return res.json()
}

export const api = {
  health: () => request('/health'),
  grades: () => request('/grades'),
  events: () => request('/events'),
  eventDetail: (id) => request(`/events/${id}`),
  live: (eventId, tMin) => request(`/live?event_id=${eventId}&t_min=${tMin}`),
  recommendations: (eventId, tMin) =>
    request(`/recommendations?event_id=${eventId}&t_min=${tMin}`),
  feedback: (advisoryId, decision, note) =>
    request(`/recommendations/${advisoryId}/feedback`, {
      method: 'POST',
      body: JSON.stringify({ decision, note }),
    }),
  correlations: () => request('/correlations'),
  stabilization: (eventId) => request(`/stabilization?event_id=${eventId}`),
  economics: () => request('/economics'),
  updateEconomics: (updates) =>
    request('/economics', { method: 'PUT', body: JSON.stringify(updates) }),
  trust: () => request('/trust'),
}
