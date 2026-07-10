"""
ML alpha layer.

Walk-forward (expanding window) validation: for each evaluation month t, the model is trained
ONLY on data strictly before t (a growing training window), then used to predict month t's
cross-section. This is refit periodically (every `REFIT_EVERY` months, not every single month,
to keep runtime sane) rather than using a single global train/test split -- a single split would
leak future regime information into early predictions and wouldn't reflect how this would
actually be deployed month over month.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

FEATURES = ["B_M", "ROE", "Leverage", "Momentum", "short_interest_stub", "vol_21d", "sharpe_21d"]
TARGET = "fwd_return_1m"
MIN_TRAIN_MONTHS = 24
REFIT_EVERY = 6

LGB_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    num_leaves=15,
    learning_rate=0.05,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    verbosity=-1,
)


def walk_forward_ml_alpha(monthly_panel: pd.DataFrame) -> pd.DataFrame:
    monthly_panel = monthly_panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    all_months = sorted(monthly_panel["date"].unique())
    if len(all_months) <= MIN_TRAIN_MONTHS:
        raise ValueError("Not enough months of history for the minimum walk-forward training window")

    predictions = []
    model = None
    for i in range(MIN_TRAIN_MONTHS, len(all_months)):
        test_month = all_months[i]
        needs_refit = model is None or (i - MIN_TRAIN_MONTHS) % REFIT_EVERY == 0
        if needs_refit:
            train_months = all_months[:i]  # strictly BEFORE test_month -- no lookahead
            train_df = monthly_panel[monthly_panel["date"].isin(train_months)]
            X_train, y_train = train_df[FEATURES], train_df[TARGET]
            model = lgb.LGBMRegressor(**LGB_PARAMS)
            model.fit(X_train, y_train)

        test_df = monthly_panel[monthly_panel["date"] == test_month]
        if test_df.empty:
            continue
        preds = model.predict(test_df[FEATURES])
        out = test_df[["date", "ticker"]].copy()
        out["ml_alpha"] = preds
        predictions.append(out)

    return pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(
        columns=["date", "ticker", "ml_alpha"]
    )


def feature_importance_report(monthly_panel: pd.DataFrame) -> dict:
    """One final fit on ALL available data purely to report feature importances for the
    attribution view -- NOT used for any prediction that feeds back into scoring, so no
    lookahead concern here (it's a diagnostic, not a forecast)."""
    X, y = monthly_panel[FEATURES], monthly_panel[TARGET]
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X, y)
    importances = dict(zip(FEATURES, model.feature_importances_.tolist()))
    total = sum(importances.values()) or 1
    return {k: v / total for k, v in importances.items()}
