"""Central configuration: paths, hyperparameters, and mappings."""

from pathlib import Path
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# ---------------------------------------------------------------------------
# Rating mappings
# ---------------------------------------------------------------------------
# Fine-grained numeric scale (AAA = 21 … C = 1)
RATING_TO_NUM: dict[str, int] = {
    "AAA": 21, "AA+": 20, "AA": 19, "AA-": 18,
    "A+": 17, "A": 16, "A-": 15,
    "BBB+": 14, "BBB": 13, "BBB-": 12,
    "BB+": 11, "BB": 10, "BB-": 9,
    "B+": 8, "B": 7, "B-": 6,
    "CCC+": 5, "CCC": 4, "CCC-": 3,
    "CC": 2, "C": 1,
}

# Seven broad buckets used for ordered probit
RATING_BUCKETS: dict[str, int] = {
    "AAA/AA": 7, "A": 6, "BBB": 5, "BB": 4,
    "B": 3, "CCC": 2, "CC-C": 1,
}

FINE_TO_BUCKET: dict[str, str] = {
    "AAA": "AAA/AA", "AA+": "AAA/AA", "AA": "AAA/AA", "AA-": "AAA/AA",
    "A+": "A", "A": "A", "A-": "A",
    "BBB+": "BBB", "BBB": "BBB", "BBB-": "BBB",
    "BB+": "BB", "BB": "BB", "BB-": "BB",
    "B+": "B", "B": "B", "B-": "B",
    "CCC+": "CCC", "CCC": "CCC", "CCC-": "CCC",
    "CC": "CC-C", "C": "CC-C",
}


# ---------------------------------------------------------------------------
# Strategy hyperparameters
# ---------------------------------------------------------------------------
@dataclass
class StrategyConfig:
    """All tunable parameters in one place."""

    # --- Optimizer (§3.6) ---
    lam: float = 10.0                       # risk aversion λ
    gamma_rated: float = 0.0025             # turnover penalty – rated bonds (25 bp)
    gamma_unrated: float = 0.0075           # turnover penalty – unrated bonds (75 bp)
    w_max_bond: float = 0.03                # max weight per bond (3 %)
    w_max_issuer: float = 0.05              # max weight per issuer (5 %)
    sector_cap_mult: float = 3.0            # sector cap = mult × EW sector weight
    sector_cap_abs: float = 0.30            # absolute sector cap (30 %)
    rating_cap_mult: float = 3.0            # rating cap = mult × EW bucket weight

    # --- Momentum (§3.4.1) ---
    momentum_windows: list[int] = field(default_factory=lambda: [1, 3, 6])

    # --- Carry (§3.4.3) ---
    carry_dts_terciles: int = 3

    # --- Data cleaning (§3.1.4) ---
    stale_threshold_days: int = 5
    return_winsor_lower: float = 0.01
    return_winsor_upper: float = 0.99

    # --- Factor standardisation (§3.4.7) ---
    signal_winsor_lower: float = 0.05
    signal_winsor_upper: float = 0.95

    # --- De-smoothing (§3.2.2) ---
    desmooth_lags: int = 2
    maturity_buckets: list[tuple[float, float]] = field(
        default_factory=lambda: [(1, 3), (3, 5), (5, 10)]
    )

    # --- Covariance (§3.5) ---
    cov_lookback: int = 60                  # months
    cov_min_obs: int = 24                   # minimum months for inclusion

    # --- Backtest (§3.7) ---
    burn_in_months: int = 36
    rebalance_freq: str = "M"               # 'M' = monthly, 'Q' = quarterly

    # --- Universe filters (§3.1.2) ---
    min_remaining_maturity_yrs: float = 1.0
    max_remaining_maturity_yrs: float = 10.0
    min_issue_size_msek: float = 300.0
    min_price_updates: int = 5
    price_update_window_days: int = 60
    min_return_history_months: int = 6
    min_cov_history_months: int = 24

    # --- Factor weights (§3.4.7) ---
    factor_weights: dict[str, float] = field(default_factory=lambda: {
        "momentum": 1 / 6,
        "value": 1 / 6,
        "carry": 1 / 6,
        "quality": 1 / 6,
        "low_risk": 1 / 6,
        "sentiment": 1 / 6,
    })
