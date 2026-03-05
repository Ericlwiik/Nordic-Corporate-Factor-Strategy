"""§3.7 — Walk-forward backtest engine with transaction cost model."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ncfs.config import StrategyConfig
from ncfs.data_loader import (
    load_nbp_prices,
    load_stamdata,
    load_ratings,
    load_fundamentals,
    load_equity_prices,
    load_short_interest,
)
from ncfs.universe import apply_filters, classify_bonds, get_primary_rating, get_rating_bucket
from ncfs.cleaning import clean_pipeline
from ncfs.returns import compute_excess_returns
from ncfs.desmoothing import estimate_smoothing_weights, desmooth_returns
from ncfs.shadow_rating import shadow_rating_pipeline, prepare_features
from ncfs.factors import (
    momentum_signal,
    value_signal,
    carry_signal,
    quality_signal,
    low_risk_signal,
    sentiment_signal,
    standardise_signal,
    composite_signal,
)
from ncfs.covariance import estimate_covariance
from ncfs.optimizer import optimize_portfolio, drift_weights


@dataclass
class BacktestResult:
    """Container for backtest outputs."""
    dates: list[pd.Timestamp] = field(default_factory=list)
    weights: list[np.ndarray] = field(default_factory=list)
    gross_returns: list[float] = field(default_factory=list)
    net_returns: list[float] = field(default_factory=list)
    transaction_costs: list[float] = field(default_factory=list)
    turnover: list[float] = field(default_factory=list)
    n_bonds: list[int] = field(default_factory=list)
    universe_isins: list[list[str]] = field(default_factory=list)
    shrinkage_deltas: list[float] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "date": self.dates,
            "gross_return": self.gross_returns,
            "net_return": self.net_returns,
            "transaction_cost": self.transaction_costs,
            "turnover": self.turnover,
            "n_bonds": self.n_bonds,
            "shrinkage_delta": self.shrinkage_deltas,
        }).set_index("date")


def compute_transaction_costs(
    w_new: np.ndarray,
    w_prev_drifted: np.ndarray,
    cost_vec: np.ndarray,
) -> float:
    """Transaction cost model (Eq. 3.17): TC = sum |Δw| * c_i / 2."""
    return np.sum(np.abs(w_new - w_prev_drifted) * cost_vec / 2)


def run_backtest(cfg: StrategyConfig | None = None) -> BacktestResult:
    """Execute the full walk-forward backtest.

    Pipeline per rebalancing date t:
    1. Load data available up to t
    2. Apply universe filters
    3. Clean prices
    4. Compute excess returns
    5. De-smooth returns
    6. Compute shadow ratings (quarterly)
    7. Construct factor signals
    8. Estimate covariance
    9. Optimise portfolio
    10. Record results
    """
    if cfg is None:
        cfg = StrategyConfig()

    # Load all data
    nbp = load_nbp_prices()
    stamdata = load_stamdata()
    ratings = load_ratings()
    fundamentals = load_fundamentals()
    equity_prices = load_equity_prices()
    short_interest = load_short_interest()

    # Determine rebalancing dates
    all_dates = pd.to_datetime(nbp["date"].unique())
    month_ends = (
        pd.Series(all_dates)
        .dt.to_period("M")
        .drop_duplicates()
        .dt.to_timestamp("M")
        .sort_values()
    )

    # Skip burn-in
    if len(month_ends) <= cfg.burn_in_months:
        raise ValueError("Insufficient data for burn-in period.")
    rebal_dates = month_ends.iloc[cfg.burn_in_months:].tolist()

    result = BacktestResult()
    w_prev = None
    prev_isins = None
    shadow_cache = {}
    shadow_quarter = None

    for t in rebal_dates:
        # 1. Universe construction
        eligible = apply_filters(nbp, stamdata, ratings, t, cfg)
        if len(eligible) < 10:
            continue
        isins = sorted(eligible)
        bond_cls = classify_bonds(stamdata, ratings, t)
        bond_cls = bond_cls.loc[bond_cls.index.isin(isins)]

        # 2. Clean prices for eligible bonds
        nbp_eligible = nbp[nbp["isin"].isin(isins) & (nbp["date"] <= t)].copy()
        nbp_clean = clean_pipeline(nbp_eligible, cfg)

        # 3. Compute excess returns
        excess_rets = compute_excess_returns(nbp_clean, bond_cls)

        # 4. De-smooth (estimate smoothing weights from full history)
        thetas = estimate_smoothing_weights(
            excess_rets, bond_cls, k=cfg.desmooth_lags
        )
        excess_rets = desmooth_returns(excess_rets, thetas, bond_cls)

        # 5. Shadow ratings (quarterly update)
        current_q = pd.Timestamp(t).to_period("Q")
        if shadow_quarter != current_q:
            shadow_ratings = shadow_rating_pipeline(
                fundamentals, equity_prices, ratings, bond_cls, t, cfg
            )
            shadow_cache = shadow_ratings.to_dict()
            shadow_quarter = current_q

        # 6. Prepare data for factor construction
        latest_nbp = nbp_clean[nbp_clean["date"] == t].set_index("isin")

        spreads = pd.Series(dtype=float)
        for isin in isins:
            if isin not in latest_nbp.index:
                continue
            row = latest_nbp.loc[isin] if isin in latest_nbp.index else None
            if row is None:
                continue
            cc = bond_cls.loc[isin, "coupon_class"] if isin in bond_cls.index else "FIXED"
            spread_val = row.get("dm", np.nan) if cc == "FRN" else row.get("z_spread", np.nan)
            if isinstance(spread_val, pd.Series):
                spread_val = spread_val.iloc[0]
            spreads[isin] = spread_val

        credit_durs = latest_nbp["credit_duration"] if "credit_duration" in latest_nbp.columns else pd.Series(dtype=float)
        maturities = pd.Series(dtype=float)  # would come from stamdata
        sectors = stamdata.set_index("isin")["sector"].reindex(isins) if "sector" in stamdata.columns else pd.Series(dtype=str)

        # Get rating for each bond (actual or shadow)
        bond_ratings = pd.Series(dtype=str)
        latest_rat = ratings[ratings["date"] <= t].sort_values("date").groupby("issuer_id").last()
        isin_to_issuer = stamdata.set_index("isin")["issuer_id"]
        for isin in isins:
            issuer = isin_to_issuer.get(isin)
            if issuer and issuer in latest_rat.index:
                fine = get_primary_rating(latest_rat.loc[issuer])
                bucket = get_rating_bucket(fine)
                bond_ratings[isin] = bucket if bucket else "unrated"
            elif issuer and issuer in shadow_cache:
                bond_ratings[isin] = str(shadow_cache[issuer])
            else:
                bond_ratings[isin] = "unrated"

        # 7. Construct factor signals
        issuer_map = isin_to_issuer.reindex(isins)
        signals = {}

        # Momentum
        mom = momentum_signal(equity_prices, t, cfg.momentum_windows)
        mom_by_isin = issuer_map.map(mom).dropna()
        if len(mom_by_isin) > 2:
            signals["momentum"] = standardise_signal(
                mom_by_isin, cfg.signal_winsor_lower, cfg.signal_winsor_upper
            )

        # Value
        if len(spreads.dropna()) > 10:
            val = value_signal(spreads, bond_ratings, sectors, maturities)
            if len(val.dropna()) > 2:
                signals["value"] = standardise_signal(
                    val, cfg.signal_winsor_lower, cfg.signal_winsor_upper
                )

        # Carry
        if len(spreads.dropna()) > 5 and len(credit_durs.dropna()) > 5:
            car = carry_signal(spreads, credit_durs)
            if len(car.dropna()) > 2:
                signals["carry"] = standardise_signal(
                    car, cfg.signal_winsor_lower, cfg.signal_winsor_upper
                )

        # Quality
        feat = prepare_features(fundamentals, equity_prices, t, cfg)
        qual_data = feat.reindex(issuer_map.values)
        qual = quality_signal(qual_data)
        qual_by_isin = issuer_map.map(qual).dropna()
        if len(qual_by_isin) > 2:
            signals["quality"] = standardise_signal(
                qual_by_isin, cfg.signal_winsor_lower, cfg.signal_winsor_upper
            )

        # Low-risk
        if len(credit_durs.dropna()) > 2:
            lr = low_risk_signal(credit_durs.reindex(isins))
            signals["low_risk"] = standardise_signal(
                lr, cfg.signal_winsor_lower, cfg.signal_winsor_upper
            )

        # Sentiment
        latest_si = short_interest[short_interest["date"] <= t]
        if not latest_si.empty:
            si_latest = latest_si.sort_values("date").groupby("issuer_id")["short_pct_outstanding"].last()
            sent = sentiment_signal(si_latest)
            sent_by_isin = issuer_map.map(sent).dropna()
            if len(sent_by_isin) > 2:
                signals["sentiment"] = standardise_signal(
                    sent_by_isin, cfg.signal_winsor_lower, cfg.signal_winsor_upper
                )

        # Composite
        mu = composite_signal(signals, cfg.factor_weights)
        mu = mu.reindex(isins, fill_value=0.0)

        N = len(isins)

        # 8. Covariance estimation
        # Build return matrix (T x N) from de-smoothed excess returns
        pivot = excess_rets.pivot(index="date", columns="isin", values="desmoothed_return")
        pivot = pivot.reindex(columns=isins)
        lookback_start = t - pd.DateOffset(months=cfg.cov_lookback)
        ret_matrix = pivot.loc[
            (pivot.index >= lookback_start) & (pivot.index <= t)
        ].values

        Sigma, delta = estimate_covariance(
            ret_matrix,
            T_effective=ret_matrix.shape[0],
            min_obs=cfg.cov_min_obs,
        )

        # 9. Handle weight carryover
        if w_prev is None or prev_isins is None:
            w_drifted = np.ones(N) / N
        else:
            # Map previous weights to current universe
            w_drifted = np.zeros(N)
            for idx, isin in enumerate(isins):
                if isin in prev_isins:
                    old_idx = prev_isins.index(isin)
                    w_drifted[idx] = w_prev[old_idx]
            # Renormalise
            total = w_drifted.sum()
            if total > 0:
                w_drifted /= total
            else:
                w_drifted = np.ones(N) / N

        # Build turnover penalty vector (bond-specific)
        gamma_vec = np.full(N, cfg.gamma_rated)
        for idx, isin in enumerate(isins):
            if isin in bond_cls.index and bond_cls.loc[isin, "rating_status"] == "unrated":
                gamma_vec[idx] = cfg.gamma_unrated

        # Universe data for constraints
        universe_data = {
            "credit_durations": credit_durs.reindex(isins, fill_value=credit_durs.mean()).values,
            "sector_ids": sectors.reindex(isins, fill_value="Other").values,
            "rating_ids": bond_ratings.reindex(isins, fill_value="unrated").values,
            "issuer_ids": issuer_map.reindex(isins, fill_value="unknown").values,
        }

        # 10. Optimise
        w_opt = optimize_portfolio(
            mu.values, Sigma, w_drifted, gamma_vec, universe_data, cfg
        )

        # Compute period return (using next month's de-smoothed returns)
        # For the backtest record, we use this period's return
        period_rets = pivot.loc[pivot.index == t].values.flatten()
        if len(period_rets) == N:
            gross_ret = float(w_opt @ np.nan_to_num(period_rets))
        else:
            gross_ret = 0.0

        # Transaction costs
        cost_vec = np.where(gamma_vec == cfg.gamma_unrated, 0.015, 0.003)
        tc = compute_transaction_costs(w_opt, w_drifted, cost_vec)
        to = float(np.sum(np.abs(w_opt - w_drifted)))

        # Record
        result.dates.append(t)
        result.weights.append(w_opt)
        result.gross_returns.append(gross_ret)
        result.net_returns.append(gross_ret - tc)
        result.transaction_costs.append(tc)
        result.turnover.append(to)
        result.n_bonds.append(N)
        result.universe_isins.append(isins)
        result.shrinkage_deltas.append(delta)

        w_prev = w_opt
        prev_isins = isins

    return result
