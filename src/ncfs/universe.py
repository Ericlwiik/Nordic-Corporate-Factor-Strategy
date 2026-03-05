"""§3.1.2–3.1.3 — Universe construction: filtering and bond classification."""

from __future__ import annotations

import pandas as pd
import numpy as np

from ncfs.config import StrategyConfig, FINE_TO_BUCKET


def apply_filters(
    nbp: pd.DataFrame,
    stamdata: pd.DataFrame,
    ratings: pd.DataFrame,
    date: pd.Timestamp,
    cfg: StrategyConfig,
) -> set[str]:
    """Apply the six inclusion filters and return the set of eligible ISINs.

    Filters are applied in the order specified in §3.1.2:
    1. Instrument type (senior unsecured / secured corporate)
    2. Currency (SEK only)
    3. Remaining maturity
    4. Minimum issue size
    5. Liquidity (minimum NBP price updates)
    6. Data completeness (minimum return history)
    """
    sd = stamdata.copy()

    # 1. Instrument type
    allowed_types = {"corporate"}
    allowed_seniority = {"senior_unsecured", "senior_secured"}
    sd = sd[
        sd["instrument_type"].isin(allowed_types)
        & sd["seniority"].isin(allowed_seniority)
    ]

    # 2. Currency
    sd = sd[sd["currency"] == "SEK"]

    # 3. Remaining maturity
    sd["remaining_maturity"] = (
        pd.to_datetime(sd["maturity_date"]) - date
    ).dt.days / 365.25
    sd = sd[
        (sd["remaining_maturity"] >= cfg.min_remaining_maturity_yrs)
        & (sd["remaining_maturity"] <= cfg.max_remaining_maturity_yrs)
    ]

    # 4. Minimum issue size
    sd = sd[sd["issue_size"] >= cfg.min_issue_size_msek]

    eligible = set(sd["isin"])

    # 5. Liquidity — require minimum price updates in recent window
    window_start = date - pd.Timedelta(days=cfg.price_update_window_days)
    recent_nbp = nbp[(nbp["date"] >= window_start) & (nbp["date"] <= date)]
    update_counts = (
        recent_nbp.groupby("isin")["clean_price"]
        .apply(lambda s: s.notna().sum())
    )
    liquid = set(update_counts[update_counts >= cfg.min_price_updates].index)
    eligible &= liquid

    # 6. Data completeness — require minimum months of return history
    history_start = date - pd.DateOffset(months=cfg.min_return_history_months)
    history_nbp = nbp[(nbp["date"] >= history_start) & (nbp["date"] <= date)]
    months_available = (
        history_nbp.groupby("isin")["date"]
        .apply(lambda s: s.dt.to_period("M").nunique())
    )
    sufficient = set(
        months_available[
            months_available >= cfg.min_return_history_months
        ].index
    )
    eligible &= sufficient

    return eligible


def classify_bonds(
    stamdata: pd.DataFrame,
    ratings: pd.DataFrame,
    date: pd.Timestamp,
) -> pd.DataFrame:
    """Classify each bond along two dimensions: coupon structure × rating status.

    Returns a DataFrame indexed by ISIN with columns:
        coupon_class  : 'FRN' or 'FIXED'
        rating_status : 'rated' or 'unrated'
    """
    sd = stamdata[["isin", "issuer_id", "coupon_type"]].copy()
    sd["coupon_class"] = sd["coupon_type"].str.upper().map(
        lambda x: "FRN" if x == "FRN" else "FIXED"
    )

    # Determine rating status at the given date
    if ratings.empty:
        sd["rating_status"] = "unrated"
    else:
        latest_ratings = (
            ratings[ratings["date"] <= date]
            .sort_values("date")
            .groupby("issuer_id")
            .last()
            .reset_index()
        )
        has_rating = latest_ratings[
            latest_ratings[["sp_rating", "moodys_rating", "fitch_rating"]]
            .notna()
            .any(axis=1)
        ]["issuer_id"]
        sd["rating_status"] = np.where(
            sd["issuer_id"].isin(has_rating), "rated", "unrated"
        )

    return sd[["isin", "coupon_class", "rating_status"]].set_index("isin")


def get_primary_rating(row: pd.Series) -> str | None:
    """Return S&P rating if available, else Moody's, else Fitch."""
    for col in ("sp_rating", "moodys_rating", "fitch_rating"):
        if pd.notna(row.get(col)):
            return row[col]
    return None


def get_rating_bucket(fine_rating: str | None) -> str | None:
    """Map a fine-grained rating string to one of the 7 broad buckets."""
    if fine_rating is None:
        return None
    return FINE_TO_BUCKET.get(fine_rating)
