"""
Confidence-weighted value model -- the ROI engine (Decision & Value layer).

Prices a recommendation in dollars: the value of the off-spec production it
would avoid, weighted by how much the system trusts the prediction behind it,
with a P10-P90 uncertainty band. This is what lets `AdvisoryPolicy` gate
low-value or low-confidence advice before it ever reaches the operator --
deliberate alarm rationalisation, ISA-18.2 in spirit, and the mechanism
`PROJECT_LOG.md` promises will suppress the risk model's residual nuisance
alarms (`AdvisoryPolicy.min_value_usd_to_surface`).

Pricing model
-------------
An off-spec tonne is assumed reworked as broke rather than downgraded and
shipped (the conservative, defensible assumption on this machine -- see
`DEV_NOTES.md`'s known limitations). It is therefore doubly costly versus prime
product: it earns none of the mill's margin (`net_margin_per_tonne`) *and* it
must be repulped (`rework_cost_per_tonne`). The dollar value of avoiding one
off-spec tonne is the sum of both terms, not either alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from . import config as C
from .config import (
    DEFAULT_ECONOMICS,
    DEFAULT_POLICY,
    AdvisoryPolicy,
    Economics,
    Source,
)
from .grades import Grade, get_grade


def production_rate_tonnes_per_min(grade: Grade) -> float:
    """Saleable production rate at a grade's nominal operating point."""
    return grade.machine_speed * C.WIRE_WIDTH_M * grade.basis_weight / 1.0e6


def usd_per_offspec_tonne(economics: Economics = DEFAULT_ECONOMICS) -> float:
    """
    Dollar cost of one tonne running off-spec: the margin it fails to earn as
    prime product, plus the cost of reworking it back into the stock system
    instead of simply discount-selling it.
    """
    return economics.net_margin_per_tonne + economics.rework_cost_per_tonne


@dataclass
class RecommendationValue:
    """A priced, confidence-weighted recommendation."""

    point_estimate_usd: float
    low_usd: float                    # P10
    high_usd: float                   # P90
    confidence: float
    avoided_off_spec_minutes: float
    avoided_off_spec_tonnes: float
    implementation_cost_usd: float
    annualized_usd: float
    source: str
    grade_code: str

    def to_dict(self) -> dict:
        return {
            "point_estimate_usd": round(self.point_estimate_usd, 2),
            "low_usd": round(self.low_usd, 2),
            "high_usd": round(self.high_usd, 2),
            "confidence": round(self.confidence, 3),
            "avoided_off_spec_minutes": round(self.avoided_off_spec_minutes, 3),
            "avoided_off_spec_tonnes": round(self.avoided_off_spec_tonnes, 4),
            "implementation_cost_usd": round(self.implementation_cost_usd, 2),
            "annualized_usd": round(self.annualized_usd, 2),
            "source": self.source,
            "grade_code": self.grade_code,
        }


def price_recommendation(
    avoided_off_spec_minutes: float,
    grade_to: str,
    confidence: float,
    source: str = Source.RISK_MODEL,
    implementation_cost_usd: float = 0.0,
    economics: Economics = DEFAULT_ECONOMICS,
) -> RecommendationValue:
    """
    Price one recommendation.

    `avoided_off_spec_minutes` is the modelled or measured reduction in
    off-spec time the recommendation is expected to produce -- e.g. the
    difference in `EventResult.labels["off_spec_minutes"]` between a rushed
    plan and the optimizer's recommended one, or the forecast cone's
    projected excursion length if the current trend is corrected now.
    Negative values are clamped to zero: a recommendation cannot be priced
    below "no effect". `confidence` is the calling model's own confidence
    (risk model probability, forecast interval tightness, ...) and must
    already be in [0, 1] -- this function does not calibrate it.

    The P10-P90 band scales the point estimate by `economics.low_multiplier`
    / `high_multiplier` (Assumption 13 in `PROJECT_LOG.md`); which multiplier
    produces the lower bound is resolved by sign so the band is correctly
    ordered even when `implementation_cost_usd` pushes the point estimate
    negative.
    """
    if not (0.0 <= confidence <= 1.0):
        raise ValueError(f"confidence must be in [0, 1], got {confidence}")

    grade = get_grade(grade_to)
    tonnes = max(avoided_off_spec_minutes, 0.0) * production_rate_tonnes_per_min(grade)
    gross_usd = tonnes * usd_per_offspec_tonne(economics)
    point = confidence * gross_usd - implementation_cost_usd

    band = sorted((point * economics.low_multiplier, point * economics.high_multiplier))

    return RecommendationValue(
        point_estimate_usd=point,
        low_usd=band[0],
        high_usd=band[1],
        confidence=confidence,
        avoided_off_spec_minutes=avoided_off_spec_minutes,
        avoided_off_spec_tonnes=tonnes,
        implementation_cost_usd=implementation_cost_usd,
        annualized_usd=point * economics.annualisation_factor(),
        source=source,
        grade_code=grade.code,
    )


def price_plan_comparison(
    baseline_off_spec_minutes: float,
    improved_off_spec_minutes: float,
    grade_to: str,
    confidence: float,
    source: str = Source.PHYSICS_MODEL,
    implementation_cost_usd: float = 0.0,
    economics: Economics = DEFAULT_ECONOMICS,
) -> RecommendationValue:
    """
    Price the value of moving from a baseline plan to an improved one, e.g.
    the twin's simulated outcome for the as-planned ramp versus the
    optimizer's recommended one. Only improvement counts: if the "improved"
    plan is actually worse, the avoided time is clamped to zero rather than
    priced negative twice (once here, once via `implementation_cost_usd`).
    """
    avoided = max(baseline_off_spec_minutes - improved_off_spec_minutes, 0.0)
    return price_recommendation(
        avoided, grade_to, confidence, source=source,
        implementation_cost_usd=implementation_cost_usd, economics=economics,
    )


def should_surface(
    value: RecommendationValue, policy: AdvisoryPolicy = DEFAULT_POLICY
) -> bool:
    """
    Gate a priced recommendation the way `AdvisoryPolicy` intends: advice
    below the value or confidence floor is not surfaced, so the system does
    not become another nuisance-alarm source.
    """
    return (
        value.point_estimate_usd >= policy.min_value_usd_to_surface
        and value.confidence >= policy.min_confidence_to_surface
    )


def portfolio_annual_value(
    per_event_usd: Sequence[float], economics: Economics = DEFAULT_ECONOMICS
) -> float:
    """
    Extrapolate a representative sample of per-event dollar values (e.g. one
    per validation-set event) to an annual figure: mean value per transition
    times the plant's actual annual transition count.

    This is the correct way to combine many events. Summing each event's own
    `RecommendationValue.annualized_usd` would double-count -- every one of
    those already assumes *that specific* scenario recurs at the full annual
    rate, which is only true in aggregate, across the whole mix of
    transitions, not for each individually.
    """
    if not per_event_usd:
        return 0.0
    return float(np.mean(per_event_usd)) * economics.annualisation_factor()
