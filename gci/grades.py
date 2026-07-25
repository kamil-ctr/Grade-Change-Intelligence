"""
Grade / recipe library and the *known* control-loop graph.

Two jobs:

1. Define the product grades a machine runs, their quality targets, and the
   recipe constraints (actuator limits, max ramp rates). This is the
   RECIPE_LIMIT source of inference.

2. Encode the control relationships the QCS *already knows about*. The
   correlation-discovery engine flags any statistically strong relationship
   that is NOT in this graph as "novel" -- which is precisely what the
   problem statement asks for ("find new correlations not defined in the
   system").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# Tag dictionary -- process variables the twin exposes
# ---------------------------------------------------------------------------
MV_TAGS: Tuple[str, ...] = (
    "stock_flow",        # m3/min  thick stock to headbox
    "filler_flow",       # m3/min  filler slurry
    "steam_pressure",    # kPa     dryer section header
    "machine_speed",     # m/min   reel / wire speed
    "retention_aid",     # ppm     polymer dosage
    "jet_wire_ratio",    # -       headbox jet vs wire speed
)

CV_TAGS: Tuple[str, ...] = (
    "basis_weight",      # g/m2
    "moisture",          # %
    "ash",               # %
    "caliper",           # um
)

DV_TAGS: Tuple[str, ...] = (
    "headbox_consistency",   # fraction
    "broke_ratio",           # fraction of furnish from broke
    "wire_drainage_index",   # - (wire condition)
    "steam_header_kpa",      # kPa available supply
    "white_water_freeness",  # CSF ml
    "couch_vacuum",          # kPa
    "press_load",            # kN/m
    "refiner_sec",           # kWh/t specific edge load
    "ambient_humidity",      # %
)

ALL_TAGS: Tuple[str, ...] = MV_TAGS + CV_TAGS + DV_TAGS

TAG_UNITS: Dict[str, str] = {
    "stock_flow": "m3/min",
    "filler_flow": "m3/min",
    "steam_pressure": "kPa",
    "machine_speed": "m/min",
    "retention_aid": "ppm",
    "jet_wire_ratio": "-",
    "basis_weight": "g/m2",
    "moisture": "%",
    "ash": "%",
    "caliper": "um",
    "headbox_consistency": "frac",
    "broke_ratio": "frac",
    "wire_drainage_index": "-",
    "steam_header_kpa": "kPa",
    "white_water_freeness": "ml CSF",
    "couch_vacuum": "kPa",
    "press_load": "kN/m",
    "refiner_sec": "kWh/t",
    "ambient_humidity": "%",
}

TAG_LABEL: Dict[str, str] = {
    "stock_flow": "Thick stock flow",
    "filler_flow": "Filler flow",
    "steam_pressure": "Dryer steam pressure",
    "machine_speed": "Machine speed",
    "retention_aid": "Retention aid dosage",
    "jet_wire_ratio": "Jet/wire ratio",
    "basis_weight": "Basis weight",
    "moisture": "Moisture",
    "ash": "Ash content",
    "caliper": "Caliper",
    "headbox_consistency": "Headbox consistency",
    "broke_ratio": "Broke ratio",
    "wire_drainage_index": "Wire drainage index",
    "steam_header_kpa": "Steam header pressure",
    "white_water_freeness": "White water freeness",
    "couch_vacuum": "Couch vacuum",
    "press_load": "Press load",
    "refiner_sec": "Refiner specific energy",
    "ambient_humidity": "Ambient humidity",
}


# ---------------------------------------------------------------------------
# KNOWN control loops (what the existing QCS/MD control already models)
# ---------------------------------------------------------------------------
# (cause, effect) pairs. Anything strongly correlated but absent from this set
# is reported by the discovery engine as a NOVEL relationship.
KNOWN_LOOPS: Tuple[Tuple[str, str], ...] = (
    ("stock_flow", "basis_weight"),
    ("machine_speed", "basis_weight"),
    ("filler_flow", "ash"),
    ("filler_flow", "basis_weight"),
    ("steam_pressure", "moisture"),
    ("machine_speed", "moisture"),
    ("basis_weight", "moisture"),
    ("basis_weight", "caliper"),
    ("steam_header_kpa", "steam_pressure"),
)

KNOWN_LOOP_SET = frozenset(KNOWN_LOOPS)


def is_known_relationship(cause: str, effect: str) -> bool:
    """True if the QCS already accounts for this cause->effect link."""
    return (cause, effect) in KNOWN_LOOP_SET or (effect, cause) in KNOWN_LOOP_SET


# ---------------------------------------------------------------------------
# Grade definitions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Grade:
    """A product grade: quality targets plus recipe operating envelope."""

    code: str
    name: str
    basis_weight: float          # g/m2 target
    moisture: float              # % target
    ash: float                   # % target
    machine_speed: float         # m/min recipe speed

    # Recipe actuator envelope (min, max)
    stock_flow_limits: Tuple[float, float] = (5.0, 80.0)
    filler_flow_limits: Tuple[float, float] = (0.0, 1.2)
    steam_pressure_limits: Tuple[float, float] = (120.0, 1100.0)
    speed_limits: Tuple[float, float] = (300.0, 1200.0)
    retention_aid_limits: Tuple[float, float] = (100.0, 600.0)

    # Max ramp rates permitted by the recipe (per minute)
    max_stock_ramp: float = 3.0          # m3/min per min
    max_filler_ramp: float = 0.10        # m3/min per min
    max_steam_ramp: float = 60.0         # kPa per min
    max_speed_ramp: float = 45.0         # m/min per min

    # Nominal transition duration the mill currently allows
    nominal_ramp_min: float = 6.0

    @property
    def bw_band(self) -> Tuple[float, float]:
        from .config import BW_SPEC_PCT

        tol = self.basis_weight * BW_SPEC_PCT / 100.0
        return (self.basis_weight - tol, self.basis_weight + tol)

    @property
    def bw_tolerance(self) -> float:
        from .config import BW_SPEC_PCT

        return self.basis_weight * BW_SPEC_PCT / 100.0


GRADE_LIBRARY: Dict[str, Grade] = {
    g.code: g
    for g in (
        Grade("NP-45", "Newsprint 45", 45.0, 8.5, 8.0, 950.0,
              nominal_ramp_min=5.0),
        Grade("SC-56", "Supercalendered 56", 56.0, 6.2, 28.0, 880.0,
              nominal_ramp_min=6.0),
        Grade("LWC-52", "Lightweight coated base 52", 52.0, 7.4, 18.0, 900.0,
              nominal_ramp_min=6.0),
        Grade("WFU-70", "Woodfree uncoated 70", 70.0, 5.2, 14.0, 720.0,
              nominal_ramp_min=7.0),
        Grade("WFU-80", "Woodfree uncoated 80", 80.0, 5.0, 16.0, 680.0,
              nominal_ramp_min=7.0),
        Grade("BRD-120", "Folding boxboard 120", 120.0, 6.6, 10.0, 520.0,
              nominal_ramp_min=8.0),
        Grade("BRD-150", "Folding boxboard 150", 150.0, 7.0, 12.0, 450.0,
              nominal_ramp_min=9.0),
    )
}

GRADE_CODES = tuple(GRADE_LIBRARY.keys())


def get_grade(code: str) -> Grade:
    try:
        return GRADE_LIBRARY[code]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(
            f"Unknown grade '{code}'. Known grades: {', '.join(GRADE_CODES)}"
        ) from exc


def transition_magnitude(from_code: str, to_code: str) -> float:
    """
    Relative severity of a grade change, used to stratify event generation and
    as a feature. Combines basis weight, ash and speed change.
    """
    a, b = get_grade(from_code), get_grade(to_code)
    bw = abs(b.basis_weight - a.basis_weight) / max(a.basis_weight, 1e-6)
    ash = abs(b.ash - a.ash) / 30.0
    spd = abs(b.machine_speed - a.machine_speed) / max(a.machine_speed, 1e-6)
    return float(bw + 0.5 * ash + 0.5 * spd)
