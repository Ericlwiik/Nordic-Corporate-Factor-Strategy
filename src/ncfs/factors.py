"""§3.4 — Factor signal construction: momentum, value, carry, quality, low-risk, sentiment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ncfs.config import StrategyConfig


# ── Momentum (§3.4.1) ────────────────────────────────────────────────────

def momentum_signal(
    equity_prices: pd.DataFrame,
    date: pd.Timestamp,
    windows: list[int] | None = None,
) -> pd.Series:
    """Equity Momentum in Credit (EMC) composite signal (Eq. 3.8).

    Mom_{i,t} = (1/3) * sum_{k in windows} R^eq_{t-k:t} / sqrt(k)

    Parameters
    ----------
    equity_prices : DataFrame [issuer_id, date, price]
    date : rebalancing date
    windows : lookback windows in months (default [1, 3, 6])

    Returns
    -------
    Series indexed by issuer_id with the composite momentum score.
    """
    if windows is None:
        windows = [1, 3, 6]

    eq = equity_prices[equity_prices["date"] <= date].sort_values(["issuer_id", "date"])

    signals = {}
    for k in windows:
        start = date - pd.DateOffset(months=k)
        price_start = eq[eq["date"] <= start].groupby("issuer_id")["price"].last()
        price_end = eq[eq["date"] <= date].groupby("issuer_id")["price"].last()
        cum_return = (price_end / price_start) - 1
        signals[k] = cum_return / np.sqrt(k)

    combined = pd.DataFrame(signals)
    composite = combined.mean(axis=1)
    composite.name = "momentum"
    return composite


# ── Value (§3.4.2) ───────────────────────────────────────────────────────

def value_signal(
    spreads: pd.Series,
    ratings: pd.Series,
    sectors: pd.Series,
    maturities: pd.Series,
) -> pd.Series:
    """Cross-sectional value signal (Eq. 3.9–3.10).

    Regresses spread on rating dummies, sector dummies, and maturity.
    Value = (s - s_hat) / s_hat

    Parameters
    ----------
    spreads    : Series indexed by isin (DM for FRNs, Z-spread for fixed)
    ratings    : Series indexed by isin (rating bucket label or shadow rating)
    sectors    : Series indexed by isin (sector label)
    maturities : Series indexed by isin (remaining maturity in years)

    Returns
    -------
    Series indexed by isin with the value signal.
    """
    df = pd.DataFrame({
        "spread": spreads,
        "rating": ratings,
        "sector": sectors,
        "maturity": maturities,
    }).dropna(subset=["spread"])

    if len(df) < 10:
        return pd.Series(dtype=float, name="value")

    # Create dummy variables
    dummies = pd.get_dummies(df[["rating", "sector"]], drop_first=True, dtype=float)
    X = pd.concat([dummies, df[["maturity"]]], axis=1)
    X["intercept"] = 1.0
    y = df["spread"]

    # OLS
    try:
        beta = np.linalg.lstsq(X.values, y.values, rcond=None)[0]
        s_hat = X.values @ beta
    except np.linalg.LinAlgError:
        return pd.Series(dtype=float, name="value")

    s_hat_series = pd.Series(s_hat, index=df.index)
    value = (df["spread"] - s_hat_series) / s_hat_series.replace(0, np.nan)
    value.name = "value"
    return value


# ── Carry (§3.4.3) ──────────────────────────────────────────────────────

def carry_signal(
    spreads: pd.Series,
    credit_durations: pd.Series,
    n_terciles: int = 3,
) -> pd.Series:
    """Carry signal with DTS-tercile double sort.

    Ranks bonds by spread within DTS terciles to isolate carry
    from credit risk.

    Parameters
    ----------
    spreads          : Series indexed by isin (DM or Z-spread in bps)
    credit_durations : Series indexed by isin
    n_terciles       : number of DTS groups (default 3)

    Returns
    -------
    Series indexed by isin with carry score in [0, 1].
    """
    df = pd.DataFrame({
        "spread": spreads,
        "credit_duration": credit_durations,
    }).dropna()

    df["dts"] = df["credit_duration"] * df["spread"]

    # Assign DTS terciles
    df["dts_tercile"] = pd.qcut(df["dts"], n_terciles, labels=False, duplicates="drop")

    # Within each tercile, rank by spread
    df["carry_score"] = df.groupby("dts_tercile")["spread"].rank(pct=True)
    result = df["carry_score"]
    result.name = "carry"
    return result


# ── Quality (§3.4.4) ────────────────────────────────────────────────────

def quality_signal(fundamentals: pd.DataFrame) -> pd.Series:
    """Composite quality signal (Eq. 3.11).

    Qual = (1/4) * (z_lev + z_cov + z_prof + z_stab)

    Parameters
    ----------
    fundamentals : DataFrame indexed by issuer_id with columns:
        total_leverage, interest_coverage, profitability, ebitda_cv_5yr

    Returns
    -------
    Series indexed by issuer_id.
    """
    df = fundamentals.copy()

    def z_score(s: pd.Series) -> pd.Series:
        return (s - s.mean()) / s.std()

    components = {}

    # Leverage (lower is better → negate)
    if "total_leverage" in df.columns:
        components["lev"] = z_score(-df["total_leverage"])

    # Interest coverage (higher is better)
    if "interest_coverage" in df.columns:
        components["cov"] = z_score(df["interest_coverage"])

    # Profitability (higher is better)
    if "profitability" in df.columns:
        components["prof"] = z_score(df["profitability"])

    # Earnings stability (lower CV is better → negate)
    if "ebitda_cv_5yr" in df.columns:
        components["stab"] = z_score(-df["ebitda_cv_5yr"])

    if not components:
        return pd.Series(dtype=float, name="quality")

    quality = pd.DataFrame(components).mean(axis=1)
    quality.name = "quality"
    return quality


# ── Low-risk (§3.4.5) ───────────────────────────────────────────────────

def low_risk_signal(credit_durations: pd.Series) -> pd.Series:
    """Low-risk signal: negative Credit Duration (Eq. 3.12).

    Bonds with lower spread sensitivity receive higher scores.
    """
    result = -credit_durations
    result.name = "low_risk"
    return result


# ── Sentiment (§3.4.6) ──────────────────────────────────────────────────

def sentiment_signal(short_interest: pd.Series) -> pd.Series:
    """Sentiment signal: negative short interest (Eq. 3.13).

    Low short interest → high score (positive market sentiment).
    """
    result = -short_interest
    result.name = "sentiment"
    return result


# ── Standardisation & Composite (§3.4.7) ────────────────────────────────

def standardise_signal(
    raw_signal: pd.Series,
    winsor_lower: float = 0.05,
    winsor_upper: float = 0.95,
) -> pd.Series:
    """Winsorise at percentiles then compute cross-sectional z-score (Eq. 3.14)."""
    s = raw_signal.copy()
    lo = s.quantile(winsor_lower)
    hi = s.quantile(winsor_upper)
    s = s.clip(lower=lo, upper=hi)
    z = (s - s.mean()) / s.std()
    z.name = raw_signal.name
    return z


def composite_signal(
    signals: dict[str, pd.Series],
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """Weighted composite of standardised factor signals (Eq. 3.14).

    Parameters
    ----------
    signals : dict mapping factor name → standardised z-score Series
    weights : dict mapping factor name → weight (must sum to 1).
              Defaults to equal weights.

    Returns
    -------
    Series indexed by isin/issuer_id with the composite score mu_{i,t}.
    """
    if weights is None:
        n = len(signals)
        weights = {k: 1.0 / n for k in signals}

    df = pd.DataFrame(signals)
    # Fill missing signals with 0 (neutral)
    df = df.fillna(0.0)

    mu = sum(weights[f] * df[f] for f in signals if f in weights)
    mu.name = "composite_signal"
    return mu
