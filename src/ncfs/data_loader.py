"""§3.1 — Data loaders with documented expected schemas.

Each loader returns a pandas DataFrame with the columns listed in its
docstring.  Until real data files are placed in ``data/raw/``, the
functions raise ``FileNotFoundError`` with guidance on the expected
file format.
"""

from __future__ import annotations

import pandas as pd

from ncfs.config import DATA_DIR


def _load_or_raise(filename: str, description: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Expected data file not found: {path}\n"
            f"Place your {description} file at this location."
        )
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, parse_dates=["date"])
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path, parse_dates=["date"])
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file format: {suffix}")


def load_nbp_prices(filename: str = "nbp_prices.csv") -> pd.DataFrame:
    """Load NBP evaluated prices and analytics.

    Expected columns
    ----------------
    date             : datetime – observation date
    isin             : str     – bond ISIN
    clean_price      : float   – evaluated clean price
    accrued_interest : float   – accrued interest at date
    dm               : float   – discount margin (bps), FRNs
    z_spread         : float   – Z-spread (bps), fixed-rate bonds
    credit_duration  : float   – sensitivity to spread changes (years)
    mod_duration     : float   – modified duration (years)
    bid              : float   – bid price (if available)
    ask              : float   – ask price (if available)
    coupon_payment   : float   – coupon paid during period (0 if none)
    """
    return _load_or_raise(filename, "NBP prices")


def load_stamdata(filename: str = "stamdata.csv") -> pd.DataFrame:
    """Load Stamdata bond reference data.

    Expected columns
    ----------------
    isin           : str      – bond ISIN
    issuer_id      : str      – unique issuer identifier
    issuer_name    : str      – issuer legal name
    currency       : str      – denomination currency (SEK, NOK, …)
    coupon_type    : str      – 'FRN' or 'FIXED'
    coupon_rate    : float    – coupon rate (% p.a.) or quoted margin (bps)
    maturity_date  : datetime – final maturity
    issue_size     : float    – outstanding amount (MSEK)
    sector         : str      – GICS sector or equivalent
    seniority      : str      – e.g. 'senior_unsecured', 'senior_secured'
    instrument_type: str      – e.g. 'corporate', 'covered', 'sovereign'
    """
    return _load_or_raise(filename, "Stamdata reference")


def load_ratings(filename: str = "ratings.csv") -> pd.DataFrame:
    """Load credit ratings by issuer over time.

    Expected columns
    ----------------
    issuer_id     : str      – unique issuer identifier
    date          : datetime – rating effective date
    sp_rating     : str      – S&P rating (e.g. 'BBB+'), NaN if none
    moodys_rating : str      – Moody's rating, NaN if none
    fitch_rating  : str      – Fitch rating, NaN if none
    """
    return _load_or_raise(filename, "credit ratings")


def load_fundamentals(filename: str = "fundamentals.csv") -> pd.DataFrame:
    """Load issuer-level accounting data (quarterly or annual).

    Expected columns
    ----------------
    issuer_id          : str      – unique issuer identifier
    date               : datetime – reporting period end date
    ebitda             : float    – EBITDA (MSEK)
    interest_expense   : float    – interest expense (MSEK)
    revenue            : float    – net revenue (MSEK)
    lt_debt            : float    – long-term debt (MSEK)
    st_borrowings      : float    – short-term borrowings (MSEK)
    current_lt_debt    : float    – current portion of LT debt (MSEK)
    total_assets       : float    – total assets (MSEK)
    market_equity      : float    – market capitalisation (MSEK)
    """
    return _load_or_raise(filename, "fundamentals")


def load_equity_prices(filename: str = "equity_prices.csv") -> pd.DataFrame:
    """Load daily/weekly equity prices by issuer.

    Expected columns
    ----------------
    issuer_id          : str      – unique issuer identifier
    date               : datetime – trading date
    price              : float    – closing equity price (SEK)
    shares_outstanding : float    – shares outstanding
    """
    return _load_or_raise(filename, "equity prices")


def load_short_interest(filename: str = "short_interest.csv") -> pd.DataFrame:
    """Load equity short interest data (FI disclosures).

    Expected columns
    ----------------
    issuer_id            : str      – unique issuer identifier
    date                 : datetime – disclosure date
    short_pct_outstanding: float    – short interest as fraction of shares
    """
    return _load_or_raise(filename, "short interest")
