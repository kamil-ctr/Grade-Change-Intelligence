"""
Global configuration for Grade Change Intelligence (GCI).

Everything a mill engineer would want to tune lives here so that no magic
numbers are buried in the engines. Economics are deliberately explicit and
editable -- the ROI engine exposes these to the operator UI.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

# ---------------------------------------------------------------------------
# Sampling / simulation
# ---------------------------------------------------------------------------
DT_S: float = 5.0                 # control + QCS sample interval (seconds)
STEPS_PER_MIN: float = 60.0 / DT_S

# ---------------------------------------------------------------------------
# Machine geometry / furnish constants
# ---------------------------------------------------------------------------
WIRE_WIDTH_M: float = 6.0         # trimmed reel width
STOCK_CONSISTENCY: float = 0.009  # headbox consistency (fraction, ~0.9%)
FIBER_RETENTION: float = 0.95     # first-pass fiber retention
FILLER_CONSISTENCY: float = 0.30  # filler (PCC/GCC) slurry consistency
FILLER_RETENTION: float = 0.60    # filler first-pass retention

# Moisture model calibration (see twin.PaperMachineTwin for derivation)
MOISTURE_K: float = 0.02512
MOISTURE_FLOOR: float = 1.5       # asymptotic dryness limit (%)
MOISTURE_STEAM_EXP: float = 0.85  # drying capacity ~ P^0.85

# Sheet density model (kg/m3) -> caliper
BASE_SHEET_DENSITY: float = 700.0
DENSITY_ASH_COEF: float = 0.004   # per % ash above 10
DENSITY_MOISTURE_COEF: float = 0.010

# ---------------------------------------------------------------------------
# Spec definition -- the core of the problem statement
# ---------------------------------------------------------------------------
BW_SPEC_PCT: float = 2.5          # basis weight off-spec if |dev| > 2.5% of SP
MOISTURE_SPEC_ABS: float = 0.40   # % absolute
ASH_SPEC_ABS: float = 1.50        # % absolute
CALIPER_SPEC_PCT: float = 3.0

# Prediction problem framing
RISK_HORIZON_MIN: float = 10.0    # predict breach within next N minutes
SETTLE_TOL_PCT: float = 1.0       # "stabilised" = within 1% of SP ...
SETTLE_DWELL_MIN: float = 2.0     # ... continuously for 2 minutes


# ---------------------------------------------------------------------------
# Source-of-inference tags (graded deliverable 5)
# ---------------------------------------------------------------------------
class Source:
    RECIPE_LIMIT = "RECIPE_LIMIT"
    HISTORICAL_DATA = "HISTORICAL_DATA"
    PHYSICS_MODEL = "PHYSICS_MODEL"
    CORRELATION_DISCOVERY = "CORRELATION_DISCOVERY"
    OPERATOR_PRECEDENT = "OPERATOR_PRECEDENT"
    RISK_MODEL = "RISK_MODEL"

    ALL = (
        RECIPE_LIMIT,
        HISTORICAL_DATA,
        PHYSICS_MODEL,
        CORRELATION_DISCOVERY,
        OPERATOR_PRECEDENT,
        RISK_MODEL,
    )

    LABEL = {
        RECIPE_LIMIT: "Recipe / grade limit",
        HISTORICAL_DATA: "Historical transition data",
        PHYSICS_MODEL: "First-principles twin",
        CORRELATION_DISCOVERY: "Discovered correlation",
        OPERATOR_PRECEDENT: "Operator precedent",
        RISK_MODEL: "Risk model inference",
    }


# ---------------------------------------------------------------------------
# Economics -- every recommendation is priced with these (ROI engine)
# ---------------------------------------------------------------------------
@dataclass
class Economics:
    """Editable mill economics. Defaults are mid-range published values."""

    net_margin_per_tonne: float = 95.0     # USD lost margin on saleable tonne
    rework_cost_per_tonne: float = 42.0    # USD repulping broke (energy+labour)
    steam_cost_per_gj: float = 9.0         # USD/GJ
    steam_gj_per_tonne: float = 2.4        # drying energy intensity
    grade_changes_per_day: float = 3.2     # transitions per machine per day
    operating_days_per_year: float = 340.0

    # Uncertainty band applied to the point estimate (P10..P90 multipliers)
    low_multiplier: float = 0.70
    high_multiplier: float = 1.35

    def annualisation_factor(self) -> float:
        return self.grade_changes_per_day * self.operating_days_per_year

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_ECONOMICS = Economics()


# ---------------------------------------------------------------------------
# Advisory governance posture
# ---------------------------------------------------------------------------
@dataclass
class AdvisoryPolicy:
    """
    GCI runs open-loop advisory by design: it never writes to the control
    system. This mirrors how APC advisory products are commissioned and is
    what makes the solution deployable without a safety case.
    """

    mode: str = "ADVISORY"  # ADVISORY -> SUPERVISORY -> CLOSED_LOOP
    allow_control_writeback: bool = False
    # Alarm rationalisation: suppress advice worth less than this so the
    # system never becomes another nuisance-alarm source (ISA-18.2 spirit).
    min_value_usd_to_surface: float = 150.0
    min_confidence_to_surface: float = 0.35
    max_concurrent_suggestions: int = 4

    maturity_ladder: tuple = field(
        default_factory=lambda: (
            ("ADVISORY", "Operator reads and acts. No writeback. (this build)"),
            ("SUPERVISORY", "Operator one-click accepts; GCI downloads setpoints."),
            ("CLOSED_LOOP", "GCI trims trajectories inside MPC guardrails."),
        )
    )


DEFAULT_POLICY = AdvisoryPolicy()
