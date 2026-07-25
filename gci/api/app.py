"""
FastAPI service -- thin HTTP wiring over `GCIService`. All business logic
lives in `service.py`; this module's only job is request/response shaping,
so the service stays testable without spinning up a server.

Run with:
    uvicorn gci.api.app:app --reload
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .service import GCIService

ROOT = Path(__file__).resolve().parents[2]


def _sanitize(obj):
    """
    Recursively replace non-finite floats (NaN, +/-inf) with None.

    Several labels are honestly NaN in Python (`settle_min` when a transition
    never settles, `mean_confidence_accepted` before any feedback exists) --
    that is meaningful, not a bug, and the service layer keeps it that way.
    But Starlette's default JSON renderer calls `json.dumps(..., allow_nan=False)`
    for strict RFC 8259 compliance, and a browser's `JSON.parse` would reject
    a literal `NaN` token too -- so the wire format needs the standard `null`
    idiom for "no value" at the HTTP boundary, even though the Python value
    underneath stays NaN.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


class NanSafeJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return super().render(_sanitize(content))


service = GCIService(
    models_dir=ROOT / "models",
    data_dir=ROOT / "data",
    ledger_path=ROOT / "data" / "ledger.jsonl",
)

app = FastAPI(
    title="Grade Change Intelligence API",
    description="Advisory intelligence layer for automatic grade changes. Advisory only -- never writes to a control system.",
    version="0.1.0",
    default_response_class=NanSafeJSONResponse,
)

# The frontend and API are deployed on different hosts (Vercel + Render), so
# the browser's same-origin policy needs an explicit opt-in. No cookies or
# auth tokens cross this boundary (the demo has no login), so a wildcard
# origin is safe here -- CORS is not a security boundary for a public,
# unauthenticated, advisory-only read/write API. Set CORS_ORIGINS (comma
# separated) to restrict this in a less trusting deployment.
_cors_origins = os.environ.get("CORS_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _cors_origins == "*" else _cors_origins.split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FeedbackRequest(BaseModel):
    decision: str
    note: Optional[str] = None


class EconomicsUpdate(BaseModel):
    net_margin_per_tonne: Optional[float] = None
    rework_cost_per_tonne: Optional[float] = None
    steam_cost_per_gj: Optional[float] = None
    steam_gj_per_tonne: Optional[float] = None
    grade_changes_per_day: Optional[float] = None
    operating_days_per_year: Optional[float] = None
    low_multiplier: Optional[float] = None
    high_multiplier: Optional[float] = None


@app.get("/api/health")
def health():
    return service.health()


@app.get("/api/grades")
def grades():
    return service.grades()


@app.get("/api/events")
def events():
    return service.list_events()


@app.get("/api/events/{event_id}")
def event_detail(event_id: int):
    try:
        return service.event_detail(event_id)
    except KeyError:
        raise HTTPException(404, f"unknown event_id {event_id}")


@app.get("/api/live")
def live(event_id: Optional[int] = None, t_min: float = Query(default=12.0)):
    event_id = event_id if event_id is not None else service.default_event_id()
    try:
        return service.live_state(event_id, t_min)
    except KeyError:
        raise HTTPException(404, f"unknown event_id {event_id}")


@app.get("/api/recommendations")
def recommendations(event_id: Optional[int] = None, t_min: float = Query(default=12.0)):
    event_id = event_id if event_id is not None else service.default_event_id()
    try:
        advisories = service.recommendations(event_id, t_min)
    except KeyError:
        raise HTTPException(404, f"unknown event_id {event_id}")
    return [a.to_dict() for a in advisories]


@app.post("/api/recommendations/{advisory_id}/feedback")
def feedback(advisory_id: str, body: FeedbackRequest):
    try:
        return service.feedback(advisory_id, body.decision, body.note)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/correlations")
def correlations():
    return service.correlations()


@app.get("/api/stabilization")
def stabilization(event_id: Optional[int] = None):
    event_id = event_id if event_id is not None else service.default_event_id()
    try:
        return service.stabilization(event_id)
    except KeyError:
        raise HTTPException(404, f"unknown event_id {event_id}")


@app.get("/api/economics")
def get_economics():
    return service.get_economics()


@app.put("/api/economics")
def put_economics(body: EconomicsUpdate):
    return service.update_economics(**body.model_dump())


@app.get("/api/trust")
def trust():
    return service.trust_summary()
