# Deployment

Two independent services: a FastAPI backend and a static React frontend.
Config for both is already in this repository — nothing below is
hypothetical.

## Backend — Render

`render.yaml` (repo root) is a ready-to-use Render Blueprint:

```yaml
services:
  - type: web
    name: gci-api
    runtime: python
    plan: free
    buildCommand: "pip install -r requirements.txt"
    startCommand: "uvicorn gci.api.app:app --host 0.0.0.0 --port $PORT"
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.9
```

Connect the repository in the Render dashboard ("New → Blueprint", point it
at this repo) and it builds from that file directly — no manual service
configuration needed. Trained models (`models/*.joblib`) are committed to
git specifically so the build never needs a training step.

> [!NOTE]
> The free Render plan runs on a fraction of a CPU core. Correlation
> discovery (`gci/discovery.py`) sweeps ~340 tag pairs and can take over a
> minute on that tier — `gci/api/service.py` warms it in a background
> thread at startup so `/api/correlations` never blocks a live request; it
> returns an empty list until the first sweep finishes, then the cached
> result after.

## Frontend — Vercel

`frontend/vercel.json`:

```json
{
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "framework": "vite"
}
```

Import the repository in Vercel, set the project root to `frontend/`, and
set one environment variable:

| Variable | Value |
|---|---|
| `VITE_API_BASE` | `<your Render service URL>/api` |

Without it, the frontend falls back to `/api` (a same-origin relative
path), which only works when frontend and backend share a host — see
`frontend/src/lib/api.js`.

## Local (no deployment needed to demo)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn gci.api.app:app --reload        # http://localhost:8000
```

```bash
cd frontend && npm install && npm run dev   # http://localhost:5173
```

Vite proxies `/api` to `localhost:8000` in dev (`vite.config.js`), so no
environment variable is needed locally.

## CORS

The API allows all origins by default (`allow_credentials=False`, no
authentication or cookies, so a wildcard origin carries no additional
risk). Restrict it with the `CORS_ORIGINS` environment variable
(comma-separated list) if needed — see `gci/api/app.py`.

---

_Live URLs, once deployed, belong here:_

| | URL |
|---|---|
| Backend | _add your Render URL_ |
| Frontend | _add your Vercel URL_ |
