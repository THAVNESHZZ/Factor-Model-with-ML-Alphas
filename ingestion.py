"""
Ingestion layer.

Critical correctness rule: fundamentals must be merged POINT-IN-TIME. A fundamental value is only
visible to the model on/after its public `date` (release date), and stays visible (forward-filled)
until the next release. This is what `pd.merge_asof` gives us when merged on release date <= as-of date.
Getting this wrong (e.g. merging on quarter-end instead of release date) is the single most common
source of lookahead bias in factor-model pipelines, so this file is the one to audit if IC numbers
ever look "too good to be true".
"""
from pathlib import Path
import pandas as pd
import numpy as np


REQUIRED_PRICE_COLS = {"date", "ticker", "open", "high", "low", "close", "adj_close", "volume"}
REQUIRED_FUND_COLS = {"date", "ticker", "book_value", "net_income", "total_equity", "total_debt"}


def load_prices(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    missing = REQUIRED_PRICE_COLS - set(df.columns)
    if missing:
        raise ValueError(f"market_data.csv missing required columns: {missing}")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def load_fundamentals(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    missing = REQUIRED_FUND_COLS - set(df.columns)
    if missing:
        raise ValueError(f"fundamentals.csv missing required columns: {missing}")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def merge_pit(prices: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time merge: for each (ticker, date) in prices, attach the most recent fundamentals
    release with release_date <= date. Uses merge_asof per ticker (backward direction)."""
    merged_parts = []
    for tkr, pgrp in prices.groupby("ticker", sort=False):
        fgrp = fundamentals[fundamentals["ticker"] == tkr].sort_values("date")
        if fgrp.empty:
            continue
        pgrp = pgrp.sort_values("date")
        m = pd.merge_asof(
            pgrp, fgrp.drop(columns=["ticker"]),
            on="date", direction="backward",
        )
        merged_parts.append(m)
    out = pd.concat(merged_parts, ignore_index=True) if merged_parts else pd.DataFrame()
    return out


def resample_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """Downsample daily panel to month-end observations per ticker (last trading day's row of the
    month). Forward returns / factor scores are computed on this monthly grid, matching standard
    Fama-French / Alphalens convention, while momentum & vol features are computed on the daily
    series *before* this downsample so they use the full daily granularity."""
    daily = daily.copy()
    daily["month"] = daily["date"].dt.to_period("M")
    monthly = (
        daily.sort_values(["ticker", "date"])
        .groupby(["ticker", "month"], as_index=False)
        .tail(1)
        .drop(columns=["month"])
        .reset_index(drop=True)
    )
    return monthly


def build_panel(prices_path: str | Path, fundamentals_path: str | Path) -> pd.DataFrame:
    prices = load_prices(prices_path)
    fundamentals = load_fundamentals(fundamentals_path)
    panel = merge_pit(prices, fundamentals)
    return panel
