"""
Synthetic data generator matching the AGREED schema (see project README):

  market_data.csv   : date, ticker, open, high, low, close, adj_close, volume  (daily)
  fundamentals.csv  : date, ticker, book_value, net_income, total_equity, total_debt (quarterly,
                      `date` = public earnings RELEASE date, not quarter-end, to avoid lookahead bias)

NOTE: this replaces the earlier Gemini-generated synthetic panel script. That script produced a single
monthly parquet file with pre-mixed ratios and no release-date semantics, price series, or
daily granularity, so it doesn't fit the ingestion pipeline. Kept as a reference; not used for real ingestion.

Embeds a genuine (but small and noisy) latent "quality" signal so B/M, ROE, leverage and momentum have
real, discoverable predictive power for the ML alpha model -- exactly like real equity data, where each
individual signal has a weak but nonzero true IC and the forward return is dominated by noise.

Short interest is intentionally a STUB (per project decision): a placeholder column with no embedded
signal, ready to be swapped for a real point-in-time short interest feed later.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

N_TICKERS = 150
START = "2014-01-01"
END = "2023-12-31"


def generate(n_tickers: int = N_TICKERS, start: str = START, end: str = END, seed: int = 42):
    rng = np.random.default_rng(seed)
    tickers = [f"TICK{str(i).zfill(3)}" for i in range(n_tickers)]
    trading_days = pd.bdate_range(start=start, end=end)  # business days as a daily-liquid-universe proxy

    # ---- latent per-ticker characteristics (unobservable in reality; drive the fundamentals/returns) ----
    quality = rng.normal(0, 1, n_tickers)               # true quality factor -> partially shows up in ROE
    size_log = rng.normal(9, 1.2, n_tickers)             # log market cap at t0 (~ $8k to $ tens of billions)
    beta = np.clip(rng.normal(1.0, 0.35, n_tickers), 0.3, 2.2)
    base_vol = np.clip(rng.normal(0.02, 0.006, n_tickers), 0.008, 0.05)  # daily idio vol

    # ---- simulate a daily market factor return series (mild autocorrelated regimes) ----
    n_days = len(trading_days)
    regime_vol = np.abs(rng.normal(0.008, 0.003, n_days))
    market_ret = rng.normal(0.0003, 1, n_days) * regime_vol

    price_rows = []
    for i, tkr in enumerate(tickers):
        # small persistent alpha tied to latent quality, decaying/noisy -> realistic weak signal
        idio_drift = 0.00005 * quality[i]
        idio_noise = rng.normal(0, base_vol[i], n_days)
        daily_ret = beta[i] * market_ret + idio_drift + idio_noise
        price = 50 * np.exp(np.cumsum(daily_ret)) * np.exp(size_log[i] - 9)  # scale roughly by size
        close = price
        open_ = close * (1 + rng.normal(0, 0.002, n_days))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n_days)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n_days)))
        adj_close = close  # no splits/divs modeled; kept identical for simplicity, documented below
        volume = np.abs(rng.normal(1_000_000, 300_000, n_days) * (1 + 2 * np.abs(daily_ret))).astype(int)

        df = pd.DataFrame({
            "date": trading_days,
            "ticker": tkr,
            "open": open_.round(2),
            "high": high.round(2),
            "low": low.round(2),
            "close": close.round(2),
            "adj_close": adj_close.round(2),
            "volume": volume,
        })
        price_rows.append(df)

    market_df = pd.concat(price_rows, ignore_index=True)

    # ---- quarterly fundamentals, release-dated (release lag ~35-55 days after quarter end, realistic) ----
    quarter_ends = pd.date_range(start=start, end=end, freq="QE")
    fundamentals_rows = []
    for i, tkr in enumerate(tickers):
        equity0 = np.exp(size_log[i]) / 4.0
        debt0 = equity0 * np.clip(rng.normal(0.6, 0.25), 0.05, 2.5)
        n_q = len(quarter_ends)
        # equity/net income grow with a quality-linked drift + noise -> ROE and B/M carry real (weak) signal
        equity_growth = np.cumsum(rng.normal(0.01 + 0.002 * quality[i], 0.03, n_q))
        total_equity = equity0 * np.exp(equity_growth)
        roe_target = np.clip(0.10 + 0.05 * quality[i] + rng.normal(0, 0.05, n_q), -0.3, 0.5)
        net_income = roe_target * total_equity
        total_debt = total_equity * np.clip(rng.normal(0.5, 0.2, n_q), 0.05, 3.0)
        book_value = total_equity  # simplification: book value of equity == total equity here

        release_lag_days = rng.integers(35, 56, n_q)
        release_dates = quarter_ends + pd.to_timedelta(release_lag_days, unit="D")

        fdf = pd.DataFrame({
            "date": release_dates,
            "ticker": tkr,
            "book_value": book_value.round(2),
            "net_income": net_income.round(2),
            "total_equity": total_equity.round(2),
            "total_debt": total_debt.round(2),
        })
        fundamentals_rows.append(fdf)

    fundamentals_df = pd.concat(fundamentals_rows, ignore_index=True)

    return market_df.sort_values(["date", "ticker"]), fundamentals_df.sort_values(["date", "ticker"])


if __name__ == "__main__":
    market_df, fundamentals_df = generate()
    market_df.to_csv("market_data.csv", index=False)
    fundamentals_df.to_csv("fundamentals.csv", index=False)
    print(f"market_data.csv: {len(market_df):,} rows, {market_df['ticker'].nunique()} tickers")
    print(f"fundamentals.csv: {len(fundamentals_df):,} rows")
    print(f"date range: {market_df['date'].min().date()} -> {market_df['date'].max().date()}")
