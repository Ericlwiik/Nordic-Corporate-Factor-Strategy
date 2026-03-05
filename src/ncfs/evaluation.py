"""§3.7.4 — Performance diagnostics: Sharpe, IR, drawdown, turnover, hit rate."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ncfs.backtest import BacktestResult


def sharpe_ratio(returns: np.ndarray | pd.Series, annualise: bool = True) -> float:
    """Annualised Sharpe ratio (Eq. 3.18).

    SR = (mean * 12) / (std * sqrt(12))
    """
    r = np.asarray(returns)
    r = r[~np.isnan(r)]
    if len(r) < 2 or np.std(r, ddof=1) == 0:
        return 0.0
    sr = np.mean(r) / np.std(r, ddof=1)
    if annualise:
        sr *= np.sqrt(12)
    return float(sr)


def sharpe_ratio_adjusted(returns: np.ndarray | pd.Series) -> float:
    """Autocorrelation-adjusted Sharpe ratio following Lo (2002).

    Corrects the standard sqrt(T) annualisation for serial correlation.
    """
    r = np.asarray(returns)
    r = r[~np.isnan(r)]
    T = len(r)
    if T < 3:
        return 0.0

    mu = np.mean(r)
    sigma = np.std(r, ddof=1)
    if sigma == 0:
        return 0.0

    eta = mu / sigma  # monthly Sharpe

    # Compute first-order autocorrelation
    r_demeaned = r - mu
    rho1 = np.corrcoef(r_demeaned[:-1], r_demeaned[1:])[0, 1]

    # Lo (2002) adjustment for AR(1): SR_adj = eta * sqrt(q / (1 + 2*rho1*(q-1)/q ... ))
    q = 12  # annualisation factor
    # Simplified first-order adjustment
    adjustment = 1 + 2 * rho1 * (q - 1) / q
    if adjustment <= 0:
        return float(eta * np.sqrt(q))

    sr_adj = eta * np.sqrt(q / adjustment)
    return float(sr_adj)


def information_ratio(
    portfolio_returns: np.ndarray | pd.Series,
    benchmark_returns: np.ndarray | pd.Series,
) -> float:
    """Annualised information ratio: mean(active) / std(active) * sqrt(12)."""
    active = np.asarray(portfolio_returns) - np.asarray(benchmark_returns)
    active = active[~np.isnan(active)]
    if len(active) < 2 or np.std(active, ddof=1) == 0:
        return 0.0
    return float(np.mean(active) / np.std(active, ddof=1) * np.sqrt(12))


def max_drawdown(returns: np.ndarray | pd.Series) -> float:
    """Maximum drawdown from cumulative return series."""
    r = np.asarray(returns)
    r = r[~np.isnan(r)]
    cum = np.cumprod(1 + r)
    running_max = np.maximum.accumulate(cum)
    drawdowns = cum / running_max - 1
    return float(np.min(drawdowns))


def annualised_turnover(weight_changes: list[float]) -> float:
    """Annualised two-sided turnover (Eq. 3.19).

    turnover_ann = mean(monthly_turnover) * 12
    """
    if not weight_changes:
        return 0.0
    return float(np.mean(weight_changes) * 12)


def hit_rate(active_returns: np.ndarray | pd.Series) -> float:
    """Fraction of months with positive active return."""
    r = np.asarray(active_returns)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return 0.0
    return float(np.mean(r > 0))


def tracking_error(
    portfolio_returns: np.ndarray | pd.Series,
    benchmark_returns: np.ndarray | pd.Series,
) -> float:
    """Annualised tracking error."""
    active = np.asarray(portfolio_returns) - np.asarray(benchmark_returns)
    active = active[~np.isnan(active)]
    if len(active) < 2:
        return 0.0
    return float(np.std(active, ddof=1) * np.sqrt(12))


def performance_summary(
    result: BacktestResult,
    benchmark_returns: np.ndarray | pd.Series | None = None,
) -> pd.DataFrame:
    """Compute all performance metrics and return a summary table."""
    net = np.array(result.net_returns)
    gross = np.array(result.gross_returns)

    metrics = {
        "Annualised Return (Gross)": float(np.mean(gross) * 12),
        "Annualised Return (Net)": float(np.mean(net) * 12),
        "Annualised Volatility (Net)": float(np.std(net, ddof=1) * np.sqrt(12)),
        "Sharpe Ratio (Net)": sharpe_ratio(net),
        "Sharpe Ratio Adjusted (Net)": sharpe_ratio_adjusted(net),
        "Max Drawdown (Net)": max_drawdown(net),
        "Annualised Turnover": annualised_turnover(result.turnover),
        "Avg Transaction Cost (Monthly)": float(np.mean(result.transaction_costs)),
        "Avg N Bonds": float(np.mean(result.n_bonds)),
        "Avg Shrinkage Intensity": float(np.mean(result.shrinkage_deltas)),
    }

    if benchmark_returns is not None:
        bench = np.asarray(benchmark_returns)
        active = net[:len(bench)] - bench[:len(net)]
        metrics["Information Ratio"] = information_ratio(net, bench)
        metrics["Tracking Error"] = tracking_error(net, bench)
        metrics["Hit Rate"] = hit_rate(active)

    return pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])
