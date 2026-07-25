"""
Service layer: every Phase-1 engine wrapped behind a small set of plain
Python methods the API (or a test) can call directly, independent of HTTP.
All business logic lives here; `app.py`'s only job is request/response
shaping, so this layer stays fully testable without spinning up a server.

Demo-cannot-fail posture: the risk and forecast model artefacts are loaded
best-effort at startup. If either is missing or fails to load (a clean
checkout that skipped training, say), the corresponding fields are simply
omitted from responses rather than raising -- `GCIService.health()` reports
which mode the service is actually running in, honestly, rather than hiding
the degradation. Grades, events, correlations and stabilization never depend
on the trained artefacts at all, so they keep working regardless.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import pandas as pd

from ..config import DEFAULT_ECONOMICS, DEFAULT_POLICY, AdvisoryPolicy, Economics
from ..control import ControlPlan
from ..discovery import discover_correlations, series_by_tag_from_dataset, series_by_tag_from_events
from ..events import EventResult, load_dataset
from ..forecast import forecast_cone_from_bundle, load_forecast_bundle
from ..grades import GRADE_LIBRARY
from ..ledger import AdvisoryLedger
from ..ml.explain import _predict_proba, compute_shap
from ..optimizer import PHYSICS_MODEL_CONFIDENCE, recommend_plan
from ..provenance import (
    Advisory,
    advisory_from_optimization,
    advisory_from_risk_prediction,
    rank_and_gate,
)
from ..stabilization import rank_loop_impact
from .datasource import DataSource, SimulatedDataSource


def _plan_from_meta(plan: Dict[str, float]) -> ControlPlan:
    return ControlPlan(
        ramp_min=float(plan["ramp_min"]),
        lead_scale=float(plan["lead_scale"]),
        tau_c_scale=float(plan["tau_c_scale"]),
        trim_enabled=bool(plan["trim_enabled"]),
        start_min=float(plan["start_min"]),
    )


class GCIService:
    def __init__(
        self,
        models_dir: Path,
        data_dir: Path,
        ledger_path: Optional[Path] = None,
        datasource: Optional[DataSource] = None,
    ):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.ledger_path = Path(ledger_path) if ledger_path else None
        self.economics: Economics = DEFAULT_ECONOMICS
        self.policy: AdvisoryPolicy = DEFAULT_POLICY
        self.ledger: AdvisoryLedger = (
            AdvisoryLedger.load(self.ledger_path) if self.ledger_path else AdvisoryLedger()
        )

        self.risk_bundle: Optional[dict] = None
        self.risk_load_error: Optional[str] = None
        self.forecast_bundle: Optional[dict] = None
        self.forecast_load_error: Optional[str] = None
        self._correlations_cache: Optional[List[dict]] = None

        self.datasource: DataSource = datasource or SimulatedDataSource()
        self._load_models()

    def _load_models(self) -> None:
        try:
            self.risk_bundle = joblib.load(self.models_dir / "risk_model.joblib")
        except Exception as exc:
            self.risk_load_error = f"{exc.__class__.__name__}: {exc}"

        try:
            self.forecast_bundle = load_forecast_bundle(self.models_dir / "forecast_model.joblib")
        except Exception as exc:
            self.forecast_load_error = f"{exc.__class__.__name__}: {exc}"

    # -- health --------------------------------------------------------
    def health(self) -> dict:
        return {
            "status": "ok",
            "risk_model_loaded": self.risk_bundle is not None,
            "risk_model_error": self.risk_load_error,
            "forecast_model_loaded": self.forecast_bundle is not None,
            "forecast_model_error": self.forecast_load_error,
            "n_demo_events": len(self.datasource.list_events()),
            "advisory_policy": dataclasses.asdict(self.policy),
        }

    # -- grades / events -------------------------------------------------
    def grades(self) -> List[dict]:
        return [dataclasses.asdict(g) for g in GRADE_LIBRARY.values()]

    def list_events(self) -> List[dict]:
        out = []
        for eid in self.datasource.list_events():
            ev = self.datasource.get_event(eid)
            out.append({
                "event_id": ev.event_id,
                "from_grade": ev.from_grade,
                "to_grade": ev.to_grade,
                "off_spec": bool(ev.labels["off_spec"]),
                "off_spec_minutes": ev.labels["off_spec_minutes"],
                "max_abs_dev_pct": ev.labels["max_abs_dev_pct"],
                "settle_min": ev.labels["settle_min"],
                "primary_cause": ev.faults[0]["code"] if ev.faults else "NONE",
            })
        return out

    def event_detail(self, event_id: int) -> dict:
        ev = self.datasource.get_event(event_id)
        return {
            "event_id": ev.event_id,
            "from_grade": ev.from_grade,
            "to_grade": ev.to_grade,
            "plan": ev.plan,
            "faults": ev.faults,
            "labels": ev.labels,
            "t_min": ev.t_min.tolist(),
            "series": {k: v.tolist() for k, v in ev.series.items()},
        }

    def default_event_id(self) -> int:
        return self.datasource.default_event_id()  # type: ignore[attr-defined]

    # -- live state / forecast -------------------------------------------
    def live_state(self, event_id: int, t_min: float) -> dict:
        ev = self.datasource.get_event(event_id)
        row = self.datasource.row_at(event_id, t_min)

        out: dict = {
            "event_id": event_id,
            "from_grade": ev.from_grade,
            "to_grade": ev.to_grade,
            "t_min": float(row["t_min"]),
            "basis_weight_dev_pct": float(row["bw_dev_pct"]),
            "currently_off_spec": bool(row["currently_off_spec"]),
        }

        if self.risk_bundle is not None:
            X = row[self.risk_bundle["features"]].to_frame().T.astype(float)
            proba = float(_predict_proba(self.risk_bundle["estimator"], X)[0])
            out["risk_probability"] = proba
            out["risk_threshold"] = self.risk_bundle["threshold"]
            out["risk_alarm"] = proba >= self.risk_bundle["threshold"]

        if self.forecast_bundle is not None:
            Xf = row[self.forecast_bundle["features"]].to_frame().T.astype(float)
            cone = forecast_cone_from_bundle(self.forecast_bundle, Xf)
            out["forecast_cone"] = {
                f"{h:g}min": {str(q): float(v[0]) for q, v in qs.items()}
                for h, qs in cone.items()
            }

        return out

    # -- recommendations ---------------------------------------------------
    def recommendations(self, event_id: int, t_min: float) -> List[Advisory]:
        ev = self.datasource.get_event(event_id)
        row = self.datasource.row_at(event_id, t_min)
        advisories: List[Advisory] = []

        if self.risk_bundle is not None:
            X = row[self.risk_bundle["features"]].to_frame().T.astype(float)
            proba = float(_predict_proba(self.risk_bundle["estimator"], X)[0])
            top_shap = []
            try:
                shap_result = compute_shap(self.risk_bundle["estimator"], X, max_rows=1)
                top_shap = [(d["feature"], d["shap_value"]) for d in shap_result.local(0)]
            except Exception:
                pass
            advisories.append(
                advisory_from_risk_prediction(
                    f"risk-{event_id}-{int(row['sample_idx'])}", proba, top_shap,
                )
            )

        baseline_plan = _plan_from_meta(ev.plan)
        opt = recommend_plan(
            ev.from_grade, ev.to_grade, baseline_plan, faults=(), seed=event_id,
            n_per_dim=5, n_rounds=1,
        )
        if opt.improvement.get("off_spec_minutes", 0.0) > 1e-6:
            value = opt.price(economics=self.economics)
            advisories.append(
                advisory_from_optimization(
                    f"opt-{event_id}", opt, confidence=PHYSICS_MODEL_CONFIDENCE, value=value,
                )
            )

        gated = rank_and_gate(advisories, self.policy)
        for a in gated:
            self.ledger.record(a, event_id=event_id)
        if self.ledger_path and gated:
            self.ledger.save(self.ledger_path)
        return gated

    def feedback(self, advisory_id: str, decision: str, note: Optional[str] = None) -> dict:
        self.ledger.respond(advisory_id, decision, note=note)
        if self.ledger_path:
            self.ledger.save(self.ledger_path)
        return self.ledger.evaluate()

    # -- correlations / stabilization ---------------------------------------
    def correlations(
        self, max_events: int = 200, max_lag_min: float = 3.0,
        min_abs_correlation: float = 0.35,
    ) -> List[dict]:
        if self._correlations_cache is None:
            try:
                cube, tags, meta = load_dataset(self.data_dir)
                series = series_by_tag_from_dataset(
                    cube, tags, meta, max_events=max_events, seed=1,
                )
            except Exception:
                # No persisted corpus on disk (a clean checkout that hasn't
                # run generate_data.py yet) -- fall back to the always-present
                # demo events rather than failing the endpoint.
                series = series_by_tag_from_events(
                    list(self.datasource.events.values())  # type: ignore[attr-defined]
                )
            results = discover_correlations(
                series, max_lag_min=max_lag_min, min_abs_correlation=min_abs_correlation,
            )
            self._correlations_cache = [r.to_dict() for r in results]
        return self._correlations_cache

    def stabilization(self, event_id: int) -> List[dict]:
        ev = self.datasource.get_event(event_id)
        baseline_plan = _plan_from_meta(ev.plan)
        impacts = rank_loop_impact(
            ev.from_grade, ev.to_grade, baseline_plan, faults=(), seed=event_id,
        )
        return [imp.to_dict() for imp in impacts]

    # -- economics / trust ---------------------------------------------------
    def get_economics(self) -> dict:
        return self.economics.to_dict()

    def update_economics(self, **kwargs) -> dict:
        valid = {f.name for f in dataclasses.fields(Economics)}
        updates = {k: v for k, v in kwargs.items() if v is not None and k in valid}
        self.economics = dataclasses.replace(self.economics, **updates)
        return self.economics.to_dict()

    def trust_summary(self) -> dict:
        return self.ledger.evaluate()
