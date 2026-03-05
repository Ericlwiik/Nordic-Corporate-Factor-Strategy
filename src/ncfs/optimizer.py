"""§3.6 — Portfolio construction: mean-variance QP with L1 turnover penalty."""

from __future__ import annotations

import numpy as np
import cvxpy as cp

from ncfs.config import StrategyConfig


def drift_weights(
    w_prev: np.ndarray,
    returns_period: np.ndarray,
) -> np.ndarray:
    """Compute drifted weights w_{t-1}^+ after one period of market returns.

    w_i^+ = w_i * (1 + R_i) / sum_j(w_j * (1 + R_j))
    """
    w_drifted = w_prev * (1 + returns_period)
    total = w_drifted.sum()
    if total <= 0:
        return w_prev.copy()
    return w_drifted / total


def optimize_portfolio(
    mu: np.ndarray,
    Sigma: np.ndarray,
    w_prev: np.ndarray,
    gamma_vec: np.ndarray,
    universe_data: dict,
    cfg: StrategyConfig | None = None,
) -> np.ndarray:
    """Solve the mean-variance QP with L1 turnover penalty (Eq. 3.15).

    max  w'mu - lambda * w'Sigma w - gamma * ||w - w_prev||_1

    subject to:
        - Duration neutrality (Eq. 3.16a)
        - Sector caps (Eq. 3.16b)
        - Rating caps (Eq. 3.16c)
        - Bond-level caps (Eq. 3.16d)
        - Issuer-level caps (Eq. 3.16e)
        - Full investment, long-only (Eq. 3.16f)

    Parameters
    ----------
    mu           : (N,) composite signal vector
    Sigma        : (N, N) covariance matrix
    w_prev       : (N,) drifted weights from previous period
    gamma_vec    : (N,) bond-specific turnover penalty
    universe_data: dict with keys:
        'credit_durations' : (N,) array
        'sector_ids'       : (N,) array of sector labels
        'rating_ids'       : (N,) array of rating bucket labels
        'issuer_ids'       : (N,) array of issuer identifiers
    cfg : StrategyConfig

    Returns
    -------
    (N,) optimal weight vector
    """
    if cfg is None:
        cfg = StrategyConfig()

    N = len(mu)
    w = cp.Variable(N)

    # Objective: signal - risk - turnover
    signal_term = mu @ w
    risk_term = cfg.lam * cp.quad_form(w, Sigma, assume_PSD=True)
    turnover_term = gamma_vec @ cp.abs(w - w_prev)

    objective = cp.Maximize(signal_term - risk_term - turnover_term)

    # Constraints
    constraints = []

    # Full investment + long-only
    constraints.append(cp.sum(w) == 1)
    constraints.append(w >= 0)
    constraints.append(w <= cfg.w_max_bond)

    # Duration neutrality: weighted avg credit duration = target
    credit_durs = universe_data["credit_durations"]
    ew_weights = np.ones(N) / N
    dur_target = ew_weights @ credit_durs
    constraints.append(credit_durs @ w == dur_target)

    # Sector caps
    sector_ids = universe_data["sector_ids"]
    unique_sectors = np.unique(sector_ids)
    for sector in unique_sectors:
        mask = (sector_ids == sector).astype(float)
        ew_sector_wt = mask.sum() / N
        cap = min(cfg.sector_cap_mult * ew_sector_wt, cfg.sector_cap_abs)
        constraints.append(mask @ w <= cap)

    # Rating caps
    rating_ids = universe_data["rating_ids"]
    unique_ratings = np.unique(rating_ids)
    for rating in unique_ratings:
        mask = (rating_ids == rating).astype(float)
        ew_rating_wt = mask.sum() / N
        cap = cfg.rating_cap_mult * ew_rating_wt
        constraints.append(mask @ w <= cap)

    # Issuer caps
    issuer_ids = universe_data["issuer_ids"]
    unique_issuers = np.unique(issuer_ids)
    for issuer in unique_issuers:
        mask = (issuer_ids == issuer).astype(float)
        constraints.append(mask @ w <= cfg.w_max_issuer)

    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)

    if problem.status in ("optimal", "optimal_inaccurate"):
        return w.value
    else:
        # Fallback to equal weights if solver fails
        return ew_weights
