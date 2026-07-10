"""
Feature engineering.

KNOWN SCHEMA GOTCHA (flagging explicitly rather than hiding it): the agreed fundamentals schema
gives absolute `total_equity` / `book_value` but the price file only has per-share price, with no
`shares_outstanding` column. Real Book-to-Market needs book equity / MARKET equity (price * shares
outstanding). Without shares outstanding we cannot compute true market cap, so this demo uses
`adj_close` directly as a market-value proxy for the B/M ratio -- fine for cross-sectional ranking
in a synthetic demo (everything is relative), but WRONG for production. Before using real data:
add `shares_outstanding` to fundamentals.csv (or source market cap directly from your price vendor)
and replace `market_value_proxy` below with the real market cap.
"""
import numpy as np
import pandas as pd


def compute_daily_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Adds daily-granularity features (rolling vol, rolling Sharpe) computed per ticker BEFORE
    any monthly downsampling, since these need the full daily return series to be meaningful."""
    panel = panel.sort_values(["ticker", "date"]).copy()
    grouped_close = panel.groupby("ticker")["adj_close"]
    panel["daily_return"] = grouped_close.pct_change()
    grouped_ret = panel.groupby("ticker")["daily_return"]
    panel["vol_21d"] = grouped_ret.transform(lambda s: s.rolling(21, min_periods=10).std())
    roll_mean = grouped_ret.transform(lambda s: s.rolling(21, min_periods=10).mean())
    panel["sharpe_21d"] = (roll_mean / panel["vol_21d"].replace(0, np.nan)) * np.sqrt(252)
    return panel


def compute_ratios(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["market_value_proxy"] = panel["adj_close"]  # see module docstring: proxy, not true market cap
    panel["B_M"] = panel["book_value"] / panel["market_value_proxy"].replace(0, np.nan)
    panel["ROE"] = panel["net_income"] / panel["total_equity"].replace(0, np.nan)
    panel["Leverage"] = panel["total_debt"] / panel["total_equity"].replace(0, np.nan)
    return panel


def add_short_interest_stub(panel: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """STUB per project decision: no real short-interest feed available. Placeholder column with
    no embedded signal (small uncorrelated noise), present purely so the ML feature contract
    already includes the slot -- swap this for a real point-in-time short-interest feed later
    without touching the rest of the pipeline."""
    rng = np.random.default_rng(seed)
    panel = panel.copy()
    panel["short_interest_stub"] = np.clip(rng.normal(0.03, 0.02, len(panel)), 0.0, 0.4)
    return panel


def resample_monthly(daily_panel: pd.DataFrame) -> pd.DataFrame:
    daily_panel = daily_panel.copy()
    daily_panel["month"] = daily_panel["date"].dt.to_period("M")
    monthly = (
        daily_panel.sort_values(["ticker", "date"])
        .groupby(["ticker", "month"], as_index=False)
        .tail(1)
        .drop(columns=["month"])
        .reset_index(drop=True)
    )
    return monthly


def compute_momentum(monthly_panel: pd.DataFrame) -> pd.DataFrame:
    """Standard 12m-1m momentum: return from 12 months ago to 1 month ago, EXCLUDING the most
    recent month (to avoid the well-documented short-term reversal effect contaminating the
    signal). Computed on the monthly grid, one row per ticker per month."""
    monthly_panel = monthly_panel.sort_values(["ticker", "date"]).copy()
    grouped_px = monthly_panel.groupby("ticker")["adj_close"]
    px_1m_ago = grouped_px.shift(1)
    px_12m_ago = grouped_px.shift(12)
    monthly_panel["Momentum"] = px_1m_ago / px_12m_ago - 1.0
    return monthly_panel


def compute_forward_return(monthly_panel: pd.DataFrame) -> pd.DataFrame:
    """Forward 1-month return = the label the model predicts. Computed as next month's close vs
    this month's close, then SHIFTED so it lines up with the current row's feature values -- i.e.
    the label at row t is the return realized AFTER t, never a return already known at t.
    The final month per ticker has no forward return and is dropped downstream."""
    monthly_panel = monthly_panel.sort_values(["ticker", "date"]).copy()
    grouped_px = monthly_panel.groupby("ticker")["adj_close"]
    next_month_px = grouped_px.shift(-1)
    monthly_panel["fwd_return_1m"] = next_month_px / monthly_panel["adj_close"] - 1.0
    return monthly_panel


def build_feature_panel(daily_panel: pd.DataFrame) -> pd.DataFrame:
    daily_panel = compute_daily_features(daily_panel)
    daily_panel = compute_ratios(daily_panel)
    daily_panel = add_short_interest_stub(daily_panel)
    monthly = resample_monthly(daily_panel)
    monthly = compute_momentum(monthly)
    monthly = compute_forward_return(monthly)
    feature_cols = ["B_M", "ROE", "Leverage", "Momentum", "short_interest_stub", "vol_21d", "sharpe_21d"]
    monthly = monthly.dropna(subset=feature_cols + ["fwd_return_1m"]).reset_index(drop=True)
    return monthly
