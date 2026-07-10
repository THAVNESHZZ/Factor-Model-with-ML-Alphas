"""
Alphalens integration.

Alphalens wants: a `factor` Series with a (date, asset) MultiIndex, and a wide `prices` DataFrame
(index=date, columns=asset) so it can compute its own forward returns over arbitrary periods.
We feed it the monthly close-price grid and let Alphalens compute forward returns itself (rather
than reusing our own fwd_return_1m), since that's what makes its periods=(1,3,6) decay-curve
analysis and turnover diagnostics correct out of the box.
"""
import numpy as np
import pandas as pd
import alphalens as al


def build_price_matrix(monthly_panel: pd.DataFrame) -> pd.DataFrame:
    return monthly_panel.pivot(index="date", columns="ticker", values="adj_close").sort_index()


def run_alphalens(factor_series: pd.Series, price_matrix: pd.DataFrame, periods=(1, 3, 6), quantiles=5) -> dict:
    """Returns IC stats, turnover, and mean-return-by-quantile (the decay curve source) for one
    factor. `factor_series` must be a Series with a (date, ticker) MultiIndex, no NaNs."""
    factor_data = al.utils.get_clean_factor_and_forward_returns(
        factor=factor_series,
        prices=price_matrix,
        periods=periods,
        quantiles=quantiles,
        max_loss=0.6,  # synthetic universe has ragged history per ticker; tolerate some drop
    )

    ic = al.performance.factor_information_coefficient(factor_data)
    ic_summary = {
        f"period_{p}": {
            "mean_ic": float(ic[f"{p}D"].mean()),
            "ic_std": float(ic[f"{p}D"].std()),
            "ic_ir": float(ic[f"{p}D"].mean() / ic[f"{p}D"].std()) if ic[f"{p}D"].std() else None,
        }
        for p in periods
    }

    turnover = al.performance.quantile_turnover(factor_data["factor_quantile"], quantile=quantiles)
    mean_turnover = float(turnover.mean()) if not turnover.empty else None

    mean_ret_by_q, _ = al.performance.mean_return_by_quantile(factor_data, by_group=False)
    decay_curve = {
        f"period_{p}": mean_ret_by_q[f"{p}D"].to_dict() for p in periods
    }

    return {
        "ic_summary": ic_summary,
        "mean_quantile_turnover": mean_turnover,
        "decay_curve_by_quantile": decay_curve,
    }
