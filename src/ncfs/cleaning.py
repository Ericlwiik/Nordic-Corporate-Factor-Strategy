"""§3.1.4 — Data cleaning: stale detection, spread interpolation, winsorisation."""

from __future__ import annotations

import pandas as pd
import numpy as np

from ncfs.config import StrategyConfig


def detect_stale_prices(
    prices: pd.DataFrame,
    threshold_days: int = 5,
) -> pd.DataFrame:
    """Flag prices as stale when unchanged for >threshold_days consecutive days.

    Parameters
    ----------
    prices : DataFrame with columns [date, isin, clean_price]
    threshold_days : int, number of consecutive unchanged days to trigger

    Returns
    -------
    DataFrame with added boolean column 'is_stale'.
    """
    df = prices.sort_values(["isin", "date"]).copy()
    df["price_changed"] = df.groupby("isin")["clean_price"].diff().ne(0)
    # Count consecutive days with no change
    df["unchanged_streak"] = (
        df.groupby("isin")["price_changed"]
        .transform(lambda s: s.eq(False).groupby(s.ne(s.shift()).cumsum()).cumcount() + 1)
    )
    df["is_stale"] = (~df["price_changed"]) & (df["unchanged_streak"] > threshold_days)
    return df.drop(columns=["price_changed", "unchanged_streak"])


def interpolate_stale_spreads(
    prices: pd.DataFrame,
    spread_col: str = "dm",
) -> pd.DataFrame:
    """Linearly interpolate spreads over stale windows, then recompute prices.

    Requires 'is_stale' column from detect_stale_prices().
    Interpolates the spread (DM or Z-spread) and marks the price as
    interpolated.
    """
    df = prices.sort_values(["isin", "date"]).copy()
    # Set stale spread values to NaN, then interpolate
    df.loc[df["is_stale"], spread_col] = np.nan
    df[spread_col] = df.groupby("isin")[spread_col].transform(
        lambda s: s.interpolate(method="linear")
    )
    df["is_interpolated"] = prices["is_stale"]
    return df


def winsorise(
    series: pd.Series,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.Series:
    """Winsorise a Series at the given percentiles."""
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lower=lo, upper=hi)


def clean_pipeline(
    prices: pd.DataFrame,
    cfg: StrategyConfig | None = None,
) -> pd.DataFrame:
    """Run the full three-step cleaning procedure.

    1. Manual error removal (placeholder — flagging extreme returns)
    2. Stale quote detection + spread interpolation
    3. Residual winsorisation of monthly returns
    """
    if cfg is None:
        cfg = StrategyConfig()

    # Step 1: Flag extreme returns for manual inspection
    df = prices.sort_values(["isin", "date"]).copy()
    df["monthly_return"] = df.groupby("isin")["clean_price"].pct_change()
    extreme_mask = df["monthly_return"].abs() > df["monthly_return"].abs().quantile(0.999)
    df["flagged_extreme"] = extreme_mask

    # Step 2: Stale detection + interpolation
    df = detect_stale_prices(df, threshold_days=cfg.stale_threshold_days)
    # Determine which spread column to interpolate based on available data
    for spread_col in ("dm", "z_spread"):
        if spread_col in df.columns:
            df = interpolate_stale_spreads(df, spread_col=spread_col)

    # Step 3: Winsorise monthly returns
    df["monthly_return"] = df.groupby("isin")["clean_price"].pct_change()
    df["monthly_return_clean"] = winsorise(
        df["monthly_return"],
        lower=cfg.return_winsor_lower,
        upper=cfg.return_winsor_upper,
    )

    return df
