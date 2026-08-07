// In dev, Vite proxies /api to localhost (see vite.config.js) so the default
// works with no env var. In production the frontend and backend are on
// different hosts (Vercel + Render), so VITE_API_BASE must point at the
// deployed API, e.g. https://gci-api.onrender.com/api
const BASE = import.meta.env.VITE_API_BASE || '/api'

async function requestOnce(path, options, timeoutMs) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const res = await fetch(`${BASE}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      signal: controller.signal,
      ...options,
    })
    if (!res.ok) {
      const body = await res.json().catch(() => ({}))
      throw new Error(body.detail || `${res.status} ${res.statusText}`)
    }
    return await res.json()
  } catch (err) {
    if (err.name === 'AbortError') throw new Error(`timed out after ${timeoutMs / 1000}s`)
    throw err
  } finally {
    clearTimeout(timer)
  }
}

// Render's free tier cold-starts in 30-90s, so a request that lands mid-boot
// must not just hang or throw once -- it needs a timeout per attempt (so a
// stalled cold start doesn't hang the UI forever) and a few retries with
// backoff (so the request that happened to land during the boot window
// succeeds once the container is up), rather than surfacing a permanent
// error for what is actually a transient condition.
async function request(path, options = {}, { timeoutMs = 15000, retries = 2, retryDelayMs = 1500 } = {}) {
  let lastErr
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      return await requestOnce(path, options, timeoutMs)
    } catch (err) {
      lastErr = err
      if (attempt === retries) break
      await new Promise((resolve) => setTimeout(resolve, retryDelayMs * 2 ** attempt))
    }
  }
  throw lastErr
}

// /api/health is the very first call of a session and the one most likely to
// land on a fully cold container, so it gets a longer per-attempt timeout
// and more retries than everything else -- worst case (~2min) comfortably
// covers a 90s cold start. Every other endpoint only fires after health has
// already succeeded, so the container is warm by the time they run and a
// shorter budget is enough.
const HEALTH_RETRY_OPTS = { timeoutMs: 25000, retries: 4, retryDelayMs: 3000 }

export const api = {
  health: () => request('/health', {}, HEALTH_RETRY_OPTS),
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
