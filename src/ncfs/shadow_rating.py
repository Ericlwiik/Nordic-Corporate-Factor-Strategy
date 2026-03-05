"""§3.3 — Shadow rating model: ordered probit for unrated issuers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel

from ncfs.config import RATING_BUCKETS, FINE_TO_BUCKET, StrategyConfig


def prepare_features(
    fundamentals: pd.DataFrame,
    equity_prices: pd.DataFrame,
    date: pd.Timestamp,
    cfg: StrategyConfig | None = None,
) -> pd.DataFrame:
    """Construct the six explanatory variables for the shadow rating model (§3.3.1).

    Features (per issuer, lagged >= 1 quarter):
        1. Interest coverage: ln(1 + max(0, C)) where C = (EBITDA+IntExp)/IntExp
        2. Profitability: EBITDA / Revenue
        3. Long-term leverage: LT Debt / Total Assets
        4. Total debt leverage: (LT + ST + Current LT) / Total Assets
        5. Size: ln(Market Equity / CPI)  [CPI normalisation placeholder]
        6. Equity volatility: annualised std of weekly returns, cross-sectionally standardised
    """
    if cfg is None:
        cfg = StrategyConfig()

    # Use fundamentals lagged by at least one quarter
    lag_date = date - pd.DateOffset(months=3)
    fund = fundamentals[fundamentals["date"] <= lag_date].copy()
    fund = fund.sort_values("date").groupby("issuer_id").last().reset_index()

    # 1. Interest coverage
    fund["int_coverage_raw"] = (
        (fund["ebitda"] + fund["interest_expense"]) / fund["interest_expense"]
    )
    fund["interest_coverage"] = np.log1p(np.maximum(0, fund["int_coverage_raw"]))

    # 2. Profitability
    fund["profitability"] = fund["ebitda"] / fund["revenue"].replace(0, np.nan)

    # 3. Long-term leverage
    fund["lt_leverage"] = fund["lt_debt"] / fund["total_assets"].replace(0, np.nan)

    # 4. Total debt leverage
    fund["total_leverage"] = (
        (fund["lt_debt"] + fund["st_borrowings"].fillna(0) + fund["current_lt_debt"].fillna(0))
        / fund["total_assets"].replace(0, np.nan)
    )

    # 5. Size (CPI normalisation is a placeholder — set CPI = 1 for now)
    fund["size"] = np.log(fund["market_equity"].replace(0, np.nan))

    # 6. Equity volatility
    vol_date = date - pd.DateOffset(weeks=52)
    eq = equity_prices[
        (equity_prices["date"] >= vol_date) & (equity_prices["date"] <= date)
    ].copy()
    eq = eq.sort_values(["issuer_id", "date"])
    eq["weekly_return"] = eq.groupby("issuer_id")["price"].pct_change()
    eq_vol = (
        eq.groupby("issuer_id")["weekly_return"]
        .agg(["std", "count"])
        .rename(columns={"std": "eq_vol_raw", "count": "n_obs"})
    )
    eq_vol = eq_vol[eq_vol["n_obs"] >= 30]
    eq_vol["eq_vol_ann"] = eq_vol["eq_vol_raw"] * np.sqrt(52)
    # Cross-sectional standardisation
    eq_vol["equity_volatility"] = (
        (eq_vol["eq_vol_ann"] - eq_vol["eq_vol_ann"].mean()) / eq_vol["eq_vol_ann"].std()
    )

    # Merge all features
    feature_cols = [
        "issuer_id", "interest_coverage", "profitability",
        "lt_leverage", "total_leverage", "size",
    ]
    features = fund[feature_cols].copy()
    features = features.merge(
        eq_vol[["equity_volatility"]].reset_index(),
        on="issuer_id",
        how="left",
    )

    # Winsorise at 1st/99th percentiles
    numeric_cols = [c for c in features.columns if c != "issuer_id"]
    for col in numeric_cols:
        lo = features[col].quantile(0.01)
        hi = features[col].quantile(0.99)
        features[col] = features[col].clip(lower=lo, upper=hi)

    return features.set_index("issuer_id")


def fit_ordered_probit(
    X: pd.DataFrame,
    y: pd.Series,
) -> OrderedModel:
    """Fit a 7-bucket ordered probit model on rated issuers.

    Parameters
    ----------
    X : Feature matrix (issuer × features), from prepare_features()
    y : Numeric rating bucket (1–7), one per issuer

    Returns
    -------
    Fitted OrderedModel result
    """
    X_clean = X.dropna()
    y_clean = y.loc[X_clean.index]
    common = X_clean.index.intersection(y_clean.dropna().index)
    X_clean = X_clean.loc[common]
    y_clean = y_clean.loc[common]

    model = OrderedModel(y_clean, X_clean, distr="probit")
    result = model.fit(method="bfgs", disp=False)
    return result


def predict_shadow_ratings(
    model_result,
    X_unrated: pd.DataFrame,
) -> pd.Series:
    """Predict shadow rating bucket for unrated issuers.

    Returns the most probable bucket (argmax) for each issuer.
    """
    X_clean = X_unrated.dropna()
    if X_clean.empty:
        return pd.Series(dtype=float)

    probs = model_result.predict(X_clean)
    predicted_bucket = probs.values.argmax(axis=1) + 1  # 1-indexed buckets
    return pd.Series(predicted_bucket, index=X_clean.index, name="shadow_rating")


def shadow_rating_pipeline(
    fundamentals: pd.DataFrame,
    equity_prices: pd.DataFrame,
    ratings: pd.DataFrame,
    bond_classes: pd.DataFrame,
    date: pd.Timestamp,
    cfg: StrategyConfig | None = None,
) -> pd.Series:
    """End-to-end shadow rating: prepare features, fit on rated, predict unrated.

    Returns
    -------
    Series mapping issuer_id → rating bucket (1–7) for all unrated issuers.
    """
    features = prepare_features(fundamentals, equity_prices, date, cfg)

    # Build target variable for rated issuers
    latest_ratings = (
        ratings[ratings["date"] <= date]
        .sort_values("date")
        .groupby("issuer_id")
        .last()
        .reset_index()
    )
    latest_ratings["primary_rating"] = latest_ratings.apply(
        lambda row: row["sp_rating"] if pd.notna(row["sp_rating"]) else row.get("moodys_rating"),
        axis=1,
    )
    latest_ratings["bucket"] = latest_ratings["primary_rating"].map(FINE_TO_BUCKET)
    latest_ratings["bucket_num"] = latest_ratings["bucket"].map(RATING_BUCKETS)
    rated_target = latest_ratings.set_index("issuer_id")["bucket_num"].dropna()

    # Split features into rated / unrated
    rated_issuers = set(bond_classes[bond_classes["rating_status"] == "rated"].index)
    # Map ISINs to issuer_ids would be needed; for now assume features are indexed by issuer_id
    rated_features = features.loc[features.index.isin(rated_target.index)]
    unrated_ids = features.index.difference(rated_target.index)
    unrated_features = features.loc[unrated_ids]

    if rated_features.empty or len(rated_target) < 10:
        return pd.Series(dtype=float)

    model_result = fit_ordered_probit(rated_features, rated_target)
    shadow_ratings = predict_shadow_ratings(model_result, unrated_features)

    return shadow_ratings
