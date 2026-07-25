"""
Provenance and trust -- deliverable 5: every suggestion GCI surfaces is
tagged with where it came from, how confident the system is, and why, in a
form the operator can actually evaluate rather than take on faith.

This module does not generate advice; it is the common packaging every other
engine's output passes through before reaching the operator. Each upstream
engine already tags and scores its own output at the point of computation --
`RecommendationValue.source`/`confidence` in `roi.py`, `OptimizationResult.source`
in `optimizer.py`, `LoopImpact.source` in `stabilization.py`,
`CorrelationResult.source`/`is_known` in `discovery.py`, the risk model's own
probability plus exact SHAP in `ml/`. `provenance.py`'s job is to fold all of
those into one `Advisory` shape, explain each in a sentence grounded in its
own actual computation (not a templated platitude), and apply
`AdvisoryPolicy`'s surfacing rules uniformly -- a confidence floor, a value
floor for anything priced, and a cap on how many suggestions compete for the
operator's attention at once.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .config import DEFAULT_POLICY, AdvisoryPolicy, Source
from .discovery import CorrelationResult
from .optimizer import OptimizationResult
from .roi import RecommendationValue
from .stabilization import LoopImpact


@dataclass
class Advisory:
    """One suggestion, fully provenanced, ready for the operator to see."""

    id: str
    title: str
    source: str
    confidence: float
    explanation: str
    value: Optional[RecommendationValue] = None
    detail: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "source": self.source,
            "source_label": Source.LABEL.get(self.source, self.source),
            "confidence": round(self.confidence, 3),
            "explanation": self.explanation,
            "value": self.value.to_dict() if self.value is not None else None,
            "detail": self.detail,
        }


# ---------------------------------------------------------------------------
# Grounded explanations -- one per upstream source
# ---------------------------------------------------------------------------
def explain_risk_prediction(
    probability: float,
    top_shap: Sequence[Tuple[str, float]] = (),
    n_top: int = 3,
) -> str:
    """
    Human-readable "why" for a risk-model-sourced advisory, grounded in that
    specific prediction's own exact SHAP attribution (see `ml/explain.py`)
    -- not a templated statement about the model in general.
    """
    base = (
        f"Risk model estimates {probability:.0%} probability of an "
        f"off-spec breach within the horizon"
    )
    if not top_shap:
        return base + "."
    drivers = ", ".join(
        f"{feat} ({'+' if contrib >= 0 else ''}{contrib:.3f})"
        for feat, contrib in list(top_shap)[:n_top]
    )
    return f"{base}, driven mainly by {drivers}."


def explain_correlation(result: CorrelationResult) -> str:
    kind = "a known" if result.is_known else "a newly discovered"
    return (
        f"{result.cause} leads {result.effect} by {result.best_lag_min:.2f} min "
        f"(r={result.correlation:+.2f}, MI={result.mutual_information:.2f}) -- "
        f"{kind} relationship."
    )


def explain_optimization(result: OptimizationResult) -> str:
    imp = result.improvement.get("off_spec_minutes", 0.0)
    return (
        f"Adjusting the ramp from {result.baseline_plan.ramp_min:.1f} to "
        f"{result.plan.ramp_min:.1f} min (lead x{result.plan.lead_scale:.2f}, "
        f"trim tau_c x{result.plan.tau_c_scale:.2f}) is predicted to reduce "
        f"off-spec time by {imp:.2f} min on the twin."
    )


def explain_stabilization(impact: LoopImpact) -> str:
    if impact.best_direction == "none":
        return f"{impact.parameter}: neither probed direction improves settling time."
    return (
        f"{impact.parameter} {impact.best_direction} of "
        f"{abs(impact.best_delta):.2f} is predicted to cut settling time by "
        f"{impact.improvement_min:.2f} min."
    )


# ---------------------------------------------------------------------------
# Advisory builders
# ---------------------------------------------------------------------------
def advisory_from_risk_prediction(
    advisory_id: str,
    probability: float,
    top_shap: Sequence[Tuple[str, float]] = (),
    value: Optional[RecommendationValue] = None,
) -> Advisory:
    return Advisory(
        id=advisory_id,
        title="Off-spec risk warning",
        source=Source.RISK_MODEL,
        confidence=probability,
        explanation=explain_risk_prediction(probability, top_shap),
        value=value,
        detail={"probability": probability, "top_shap": list(top_shap)},
    )


def advisory_from_correlation(advisory_id: str, result: CorrelationResult) -> Advisory:
    return Advisory(
        id=advisory_id,
        title=f"{result.cause} -> {result.effect}",
        source=result.source,
        confidence=min(abs(result.correlation), 1.0),
        explanation=explain_correlation(result),
        detail=result.to_dict(),
    )


def advisory_from_optimization(
    advisory_id: str, result: OptimizationResult, confidence: float,
    value: Optional[RecommendationValue] = None,
) -> Advisory:
    return Advisory(
        id=advisory_id,
        title=f"Recommended plan for {result.baseline_plan.ramp_min:.1f} min ramp",
        source=result.source,
        confidence=confidence,
        explanation=explain_optimization(result),
        value=value,
        detail=result.to_dict(),
    )


def advisory_from_stabilization(
    advisory_id: str, impact: LoopImpact, confidence: float,
) -> Advisory:
    return Advisory(
        id=advisory_id,
        title=f"Stabilize faster: {impact.parameter}",
        source=impact.source,
        confidence=confidence,
        explanation=explain_stabilization(impact),
        detail=impact.to_dict(),
    )


# ---------------------------------------------------------------------------
# Policy gate
# ---------------------------------------------------------------------------
def rank_and_gate(
    advisories: Sequence[Advisory], policy: AdvisoryPolicy = DEFAULT_POLICY,
) -> List[Advisory]:
    """
    Apply `AdvisoryPolicy` uniformly across every advisory type: drop
    anything below the confidence floor; drop priced advisories below the
    value floor too (unpriced ones -- discovered correlations, stabilization
    sensitivities -- have no dollar figure to gate on and are judged on
    confidence alone). Sort by priced value first (highest first, unpriced
    advisories sort after all priced ones), then confidence, and cap at
    `max_concurrent_suggestions` -- the operator is never shown more than the
    policy allows competing for their attention at once.
    """
    surfaced = [
        a for a in advisories
        if a.confidence >= policy.min_confidence_to_surface
        and (a.value is None or a.value.point_estimate_usd >= policy.min_value_usd_to_surface)
    ]
    surfaced.sort(
        key=lambda a: (
            a.value.point_estimate_usd if a.value is not None else float("-inf"),
            a.confidence,
        ),
        reverse=True,
    )
    return surfaced[: policy.max_concurrent_suggestions]
