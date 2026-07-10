"""
Fama-French style baseline.

We don't have an external FF3 factor feed (out of scope for the agreed data schema), so we
CONSTRUCT the factors from the universe itself, the same way Fama-French originally did:
  - Market factor: equal-weighted average return of the universe that month.
  - SMB (Small Minus Big): return of smallest-third minus largest-third, sorted on `market_value_proxy`.
  - HML (High Minus Low): return of highest-B/M-third minus lowest-B/M-third ("value minus growth").

Each stock's baseline "classical alpha" for a month is then the RESIDUAL from regressing its
past 24 months of returns on these three factors -- i.e. the return NOT explained by market,
size or value exposure. This residual is the classical-factor score used for IC/attribution
and is the number ML alphas will be compared and combined against.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

MIN_HISTORY_MONTHS = 18


def construct_factors(monthly_panel: pd.DataFrame) -> pd.DataFrame:
    """Builds one row per month with Market/SMB/HML returns for that month, using each stock's
    OWN realized return that month, sorted by prior-known characteristics (size, B/M) as of that
    same row (fundamentals are already point-in-time as of the row's date, so no lookahead here)."""
    rows = []
    for dt, grp in monthly_panel.groupby("date"):
        grp = grp.dropna(subset=["market_value_proxy", "B_M", "fwd_return_1m"])
        if len(grp) < 15:
            continue
        mkt = grp["fwd_return_1m"].mean()  # equal-weighted market return realized over this month->next
        size_terciles = pd.qcut(grp["market_value_proxy"], 3, labels=["small", "mid", "big"], duplicates="drop")
        bm_terciles = pd.qcut(grp["B_M"], 3, labels=["low", "mid", "high"], duplicates="drop")
        grp = grp.assign(size_tercile=size_terciles, bm_tercile=bm_terciles)
        smb = grp.loc[grp.size_tercile == "small", "fwd_return_1m"].mean() - \
            grp.loc[grp.size_tercile == "big", "fwd_return_1m"].mean()
        hml = grp.loc[grp.bm_tercile == "high", "fwd_return_1m"].mean() - \
            grp.loc[grp.bm_tercile == "low", "fwd_return_1m"].mean()
        rows.append({"date": dt, "MKT": mkt, "SMB": smb, "HML": hml})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def compute_classical_alpha(monthly_panel: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """For each (ticker, date), regress that ticker's trailing `MIN_HISTORY_MONTHS` of forward
    monthly returns on the factor returns realized over the SAME trailing window, then the
    classical baseline score at date t is the model's predicted alpha for t
    (regression intercept + residual convention: here we use the fitted intercept as the
    stock's persistent classical alpha, re-estimated fresh at each date using only PAST data)."""
    monthly_panel = monthly_panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    factors_indexed = factors.set_index("date")

    out_rows = []
    for tkr, g in monthly_panel.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True)
        for i in range(len(g)):
            if i < MIN_HISTORY_MONTHS:
                continue
            window = g.iloc[i - MIN_HISTORY_MONTHS:i]  # strictly PAST months only, current row excluded
            dates = window["date"]
            f = factors_indexed.reindex(dates)
            if f.isna().any().any():
                continue
            y = window["fwd_return_1m"].values
            X = sm.add_constant(f[["MKT", "SMB", "HML"]].values)
            try:
                model = sm.OLS(y, X).fit()
            except Exception:
                continue
            alpha = model.params[0]
            betas = model.params[1:]
            out_rows.append({
                "date": g.loc[i, "date"], "ticker": tkr,
                "classical_alpha": alpha,
                "beta_mkt": betas[0], "beta_smb": betas[1], "beta_hml": betas[2],
            })
    return pd.DataFrame(out_rows)
