"""§3.2.2 — Getmansky-Lo-Makarov de-smoothing of evaluated returns."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _ma_loglik(thetas_free: np.ndarray, returns: np.ndarray) -> float:
    """Negative log-likelihood for an MA(k) model with unit-sum constraint.

    thetas_free contains theta_1, ..., theta_k.  theta_0 = 1 - sum(thetas_free).
    The innovation variance sigma^2 is concentrated out.
    """
    theta_0 = 1.0 - thetas_free.sum()
    thetas = np.concatenate([[theta_0], thetas_free])
    k = len(thetas) - 1
    T = len(returns)

    if theta_0 <= 0:
        return 1e12

    # Recover innovations via recursive filter
    eps = np.zeros(T)
    for t in range(T):
        ma_part = sum(
            thetas[j] * eps[t - j] if t - j >= 0 else 0.0 for j in range(1, k + 1)
        )
        eps[t] = (returns[t] - ma_part) / theta_0

    sigma2 = np.mean(eps**2)
    if sigma2 <= 0:
        return 1e12

    # Log-likelihood (up to constant)
    ll = -0.5 * T * np.log(sigma2) - T * np.log(abs(theta_0))
    return -ll


def estimate_smoothing_weights(
    returns: pd.DataFrame,
    bond_classes: pd.DataFrame,
    maturity_col: str = "remaining_maturity",
    maturity_buckets: list[tuple[float, float]] | None = None,
    k: int = 2,
) -> dict[str, np.ndarray]:
    """Estimate MA(k) smoothing weights per bond-class × maturity bucket.

    Parameters
    ----------
    returns : DataFrame with columns [date, isin, excess_return]
    bond_classes : DataFrame indexed by isin with columns
        [coupon_class, rating_status] and a maturity column.
    maturity_buckets : list of (lo, hi) tuples, e.g. [(1,3), (3,5), (5,10)]
    k : MA lag order (default 2)

    Returns
    -------
    dict mapping bucket label → array [theta_0, theta_1, ..., theta_k]
    """
    if maturity_buckets is None:
        maturity_buckets = [(1, 3), (3, 5), (5, 10)]

    bc = bond_classes.copy()
    if maturity_col not in bc.columns:
        bc[maturity_col] = 5.0  # placeholder if maturity not available

    results = {}

    for coupon_class in ("FRN", "FIXED"):
        for rating_status in ("rated", "unrated"):
            for lo, hi in maturity_buckets:
                bucket_label = f"{coupon_class}_{rating_status}_{lo}-{hi}y"

                # Select ISINs in this bucket
                mask = (
                    (bc["coupon_class"] == coupon_class)
                    & (bc["rating_status"] == rating_status)
                    & (bc[maturity_col] >= lo)
                    & (bc[maturity_col] < hi)
                )
                isins = bc[mask].index.tolist()
                if not isins:
                    results[bucket_label] = np.array([1.0] + [0.0] * k)
                    continue

                # Pool returns for all bonds in this bucket
                bucket_rets = returns[returns["isin"].isin(isins)]["excess_return"].dropna().values
                if len(bucket_rets) < 3 * k:
                    results[bucket_label] = np.array([1.0] + [0.0] * k)
                    continue

                # Optimise: minimise neg-loglik subject to thetas >= 0 and sum = 1
                x0 = np.full(k, 1.0 / (k + 1))
                bounds = [(0.0, 1.0)] * k
                constraints = {"type": "ineq", "fun": lambda x: 1.0 - x.sum() - 1e-6}

                res = minimize(
                    _ma_loglik,
                    x0,
                    args=(bucket_rets,),
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                )

                if res.success:
                    theta_rest = res.x
                    theta_0 = 1.0 - theta_rest.sum()
                    thetas = np.concatenate([[theta_0], theta_rest])
                else:
                    thetas = np.array([1.0] + [0.0] * k)

                results[bucket_label] = thetas

    return results


def desmooth_returns(
    returns: pd.DataFrame,
    thetas: dict[str, np.ndarray],
    bond_classes: pd.DataFrame,
    maturity_col: str = "remaining_maturity",
    maturity_buckets: list[tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Invert the smoothing to recover true returns (Eq. 3.6).

        R_t = (R^o_t - theta_1 * R_{t-1} - theta_2 * R_{t-2}) / theta_0
    """
    if maturity_buckets is None:
        maturity_buckets = [(1, 3), (3, 5), (5, 10)]

    bc = bond_classes.copy()
    if maturity_col not in bc.columns:
        bc[maturity_col] = 5.0

    df = returns.sort_values(["isin", "date"]).copy()
    df["desmoothed_return"] = np.nan

    for isin, group in df.groupby("isin"):
        if isin not in bc.index:
            df.loc[group.index, "desmoothed_return"] = group["excess_return"]
            continue

        bond_info = bc.loc[isin]
        coupon_class = bond_info["coupon_class"]
        rating_status = bond_info["rating_status"]
        mat = bond_info.get(maturity_col, 5.0)

        # Find matching bucket
        bucket_label = None
        for lo, hi in maturity_buckets:
            if lo <= mat < hi:
                bucket_label = f"{coupon_class}_{rating_status}_{lo}-{hi}y"
                break

        if bucket_label is None or bucket_label not in thetas:
            df.loc[group.index, "desmoothed_return"] = group["excess_return"]
            continue

        theta = thetas[bucket_label]
        k = len(theta) - 1
        obs_rets = group["excess_return"].values
        T = len(obs_rets)
        true_rets = np.full(T, np.nan)

        for t in range(k, T):
            ma_part = sum(theta[j] * true_rets[t - j] for j in range(1, k + 1) if not np.isnan(true_rets[t - j]))
            true_rets[t] = (obs_rets[t] - ma_part) / theta[0] if theta[0] > 0 else obs_rets[t]

        df.loc[group.index, "desmoothed_return"] = true_rets

    return df


def variance_inflation_factor(thetas: np.ndarray) -> float:
    """Compute the variance inflation factor 1 / sum(theta_j^2) (Eq. 3.7)."""
    return 1.0 / np.sum(thetas**2)
