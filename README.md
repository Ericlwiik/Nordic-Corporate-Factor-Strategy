# Nordic Corporate Factor Strategy (NCFS)

A systematic factor investing strategy for the Nordic corporate bond market, implemented as part of a master thesis at KTH Royal Institute of Technology. The framework adapts the Systematic Active Fixed Income (SAFI) methodology to the Nordic setting, constructing optimised portfolios from six bond-level factor signals.

## Overview

The strategy generates monthly factor-based signals for each bond in a filtered Nordic corporate bond universe, constructs portfolios that maximise composite factor exposure subject to risk and turnover constraints, and evaluates performance via a strict walk-forward backtest.

**Pipeline**

```
NBP data → Clean universe → Returns → De-smoothed returns → Factor signals → Covariance → Optimisation → Backtest
  §3.1        §3.1.2         §3.2.1       §3.2.2            §3.4             §3.5         §3.6          §3.7
```

## Factors

| Factor | Signal | Source |
|--------|--------|--------|
| **Momentum** | Equity Momentum in Credit (EMC) — composite of 1/3/6-month issuer equity returns scaled by √k | Dor et al. (2020) |
| **Value** | Percentage spread deviation from a fair-spread model (rating × sector × maturity OLS) | Houweling & van Zundert (2017) |
| **Carry** | Raw spread (DM or Z-spread) ranked within DTS terciles to isolate carry from credit risk | Israel et al. (2018) |
| **Quality** | Composite z-score of leverage, interest coverage, profitability, and earnings stability | Asness et al. (2019) |
| **Low-risk** | Negative Credit Duration (not Modified Duration, to correctly rank FRNs) | Monnberg (2025) |
| **Sentiment** | Negative equity short interest from FI regulatory disclosures | Furey et al. (2025) |

## Portfolio Construction

The optimiser solves a mean-variance QP with an L1 turnover penalty:

```
max  w'μ  −  λ w'Σw  −  γ ‖w − w₋₁⁺‖₁
```

Subject to duration neutrality, sector caps, rating caps, bond/issuer concentration limits, long-only, and full investment constraints. The covariance matrix uses Ledoit-Wolf shrinkage toward a constant-correlation target, estimated from de-smoothed returns.

## Project Structure

```
├── environment.yml              # Conda environment specification
├── pyproject.toml               # Package metadata (pip install -e .)
├── data/raw/                    # Place raw data files here
├── notebooks/
│   └── 00_pipeline.ipynb        # Main backtest runner notebook
├── src/ncfs/
│   ├── config.py                # All hyperparameters and constants
│   ├── data_loader.py           # Data loaders with documented schemas
│   ├── universe.py              # Inclusion filters and bond classification
│   ├── cleaning.py              # Stale price detection, interpolation, winsorisation
│   ├── returns.py               # Total and excess return computation
│   ├── desmoothing.py           # Getmansky-Lo-Makarov MA(2) de-smoothing
│   ├── shadow_rating.py         # Ordered probit model for unrated issuers
│   ├── factors.py               # Six factor signals and composite construction
│   ├── covariance.py            # Pairwise sample covariance + Ledoit-Wolf shrinkage
│   ├── optimizer.py             # CVXPY quadratic program with L1 turnover penalty
│   ├── backtest.py              # Walk-forward backtest engine
│   └── evaluation.py            # Performance metrics (Sharpe, IR, drawdown, etc.)
└── tests/
```

## Setup

```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate ncfs

# Install the package in editable mode
pip install -e .

# Launch the notebook
jupyter lab notebooks/00_pipeline.ipynb
```

## Data

The strategy requires the following data sources, placed in `data/raw/` as CSV files:

| File | Description | Key columns |
|------|-------------|-------------|
| `nbp_prices.csv` | NBP evaluated prices and analytics | `date, isin, clean_price, accrued_interest, dm, z_spread, credit_duration` |
| `stamdata.csv` | Stamdata bond reference data | `isin, issuer_id, currency, coupon_type, maturity_date, issue_size, sector` |
| `ratings.csv` | Credit ratings over time | `issuer_id, date, sp_rating, moodys_rating, fitch_rating` |
| `fundamentals.csv` | Issuer accounting data | `issuer_id, date, ebitda, interest_expense, lt_debt, total_assets` |
| `equity_prices.csv` | Issuer equity prices | `issuer_id, date, price, shares_outstanding` |
| `short_interest.csv` | Equity short interest (FI) | `issuer_id, date, short_pct_outstanding` |

Full column schemas are documented in each `load_*()` function in `src/ncfs/data_loader.py`.

## Key Design Choices

- **Bond classification**: Every pipeline step handles four bond classes explicitly (FRN/Fixed × Rated/Unrated), using DM for FRNs and Z-spread for fixed-rate bonds
- **De-smoothing**: NBP evaluated prices are smoothed by construction; Getmansky-Lo-Makarov MA(2) inversion recovers true volatility and removes serial correlation
- **Shadow ratings**: An ordered probit model assigns unrated issuers (~53% of Nordic HY) to rating buckets, enabling value factor computation and rating-based constraints across the full universe
- **Credit Duration**: Used instead of Modified Duration for the low-risk factor and duration constraint, preventing misclassification of FRNs as low-risk
- **Monthly rebalancing**: Adapted from SAFI's daily approach to reflect lower Nordic market liquidity
