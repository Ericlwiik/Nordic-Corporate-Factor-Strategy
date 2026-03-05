"""§3.2.1 — Total return and excess return computation for FRNs and fixed-rate bonds."""

from __future__ import annotations

import pandas as pd
import numpy as np


def compute_total_returns(
    prices: pd.DataFrame,
    bond_classes: pd.DataFrame,
) -> pd.DataFrame:
    """Compute monthly total returns for each bond.

    For both FRNs (Eq. 3.1) and fixed-rate bonds (Eq. 3.3):
        R_{i,t} = (P_t + AI_t + C_{[t-1,t]}) / (P_{t-1} + AI_{t-1}) - 1

    Parameters
    ----------
    prices : DataFrame with columns
        [date, isin, clean_price, accrued_interest, coupon_payment]
    bond_classes : DataFrame indexed by isin with column 'coupon_class'

    Returns
    -------
    DataFrame with columns [date, isin, total_return, coupon_class]
    """
    df = prices.sort_values(["isin", "date"]).copy()

    # Dirty price = clean + accrued
    df["dirty_price"] = df["clean_price"] + df["accrued_interest"].fillna(0)

    # Coupon payment during the period (0 if no payment)
    df["coupon_payment"] = df["coupon_payment"].fillna(0)

    # Lag the dirty price within each bond
    df["dirty_price_prev"] = df.groupby("isin")["dirty_price"].shift(1)

    # Total return: (P_t + AI_t + C) / (P_{t-1} + AI_{t-1}) - 1
    df["total_return"] = (
        (df["clean_price"] + df["accrued_interest"].fillna(0) + df["coupon_payment"])
        / df["dirty_price_prev"]
        - 1
    )

    # Merge bond class
    df = df.merge(
        bond_classes[["coupon_class"]], left_on="isin", right_index=True, how="left"
    )

    return df[["date", "isin", "total_return", "coupon_class"]].dropna(
        subset=["total_return"]
    )


def compute_excess_returns(
    prices: pd.DataFrame,
    bond_classes: pd.DataFrame,
) -> pd.DataFrame:
    """Compute monthly excess returns isolating the credit component (Eq. 3.5).

        R^excess_{i,t} ≈ -D^credit_{i,t-1} * Δs_{i,t} + carry_{i,t}

    where Δs is the change in DM (FRNs) or Z-spread (fixed-rate).

    Parameters
    ----------
    prices : DataFrame with columns
        [date, isin, dm, z_spread, credit_duration, coupon_class]
        or merge bond_classes separately.
    bond_classes : DataFrame indexed by isin with column 'coupon_class'

    Returns
    -------
    DataFrame with columns [date, isin, excess_return, coupon_class]
    """
    df = prices.sort_values(["isin", "date"]).copy()
    df = df.merge(
        bond_classes[["coupon_class"]], left_on="isin", right_index=True, how="left"
    )

    # Select the appropriate spread column per bond class
    df["spread"] = np.where(
        df["coupon_class"] == "FRN", df["dm"], df["z_spread"]
    )

    # Lagged values within each bond
    df["spread_prev"] = df.groupby("isin")["spread"].shift(1)
    df["credit_dur_prev"] = df.groupby("isin")["credit_duration"].shift(1)

    # Spread change
    df["delta_spread"] = df["spread"] - df["spread_prev"]

    # Convert spread from bps to decimal for return calc
    df["delta_spread_dec"] = df["delta_spread"] / 10_000

    # Carry component: monthly accrual of spread (spread / 12, in decimal)
    df["carry"] = df["spread_prev"] / 10_000 / 12

    # Excess return ≈ -D^credit * Δs + carry
    df["excess_return"] = -df["credit_dur_prev"] * df["delta_spread_dec"] + df["carry"]

    return df[["date", "isin", "excess_return", "coupon_class"]].dropna(
        subset=["excess_return"]
    )
