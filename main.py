import json
import time
import traceback
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.ingestion import build_panel
from app.features import build_feature_panel
from app.factor_model import construct_factors, compute_classical_alpha
from app.ml_alpha import walk_forward_ml_alpha, feature_importance_report
from app.analysis import build_price_matrix, run_alphalens

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Factor Model with ML Alphas API")

# NOTE: fixed from the original stub -- allow_origins=["*"] combined with allow_credentials=True
# is rejected by browsers per the CORS spec (wildcard origin + credentials is disallowed), so any
# real cross-origin request from the Vercel frontend would silently fail. Since this API has no
# cookie/session auth, we drop allow_credentials rather than hardcode a domain list that would
# need to track every preview deploy URL. Tighten to allow_origin_regex for your prod domain
# once you have a fixed URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_state = {"status": "idle", "error": None}


class PipelineStatus(BaseModel):
    status: str
    error: str | None = None


@app.get("/health")
async def health_check():
    return {"status": "healthy", "pipeline": _state["status"]}


@app.post("/pipeline/run", response_model=PipelineStatus)
async def run_pipeline():
    """Runs ingestion -> features -> classical factor model -> ML alpha -> Alphalens analysis,
    end to end, and caches every intermediate + final artifact to disk as parquet/json so the
    other endpoints can just read from cache instead of recomputing on every request."""
    prices_path = DATA_DIR / "market_data.csv"
    fundamentals_path = DATA_DIR / "fundamentals.csv"
    if not prices_path.exists() or not fundamentals_path.exists():
        raise HTTPException(
            status_code=400,
            detail="market_data.csv / fundamentals.csv not found in backend/data. "
                   "Upload real data via /data/upload or run data/generate_synthetic_data.py.",
        )

    _state["status"] = "running"
    _state["error"] = None
    t0 = time.time()
    try:
        panel = build_panel(prices_path, fundamentals_path)
        feat = build_feature_panel(panel)

        factors = construct_factors(feat)
        classical_alpha_df = compute_classical_alpha(feat, factors)

        ml_preds_df = walk_forward_ml_alpha(feat)
        importances = feature_importance_report(feat)

        price_matrix = build_price_matrix(feat)

        classical_ic = run_alphalens(
            classical_alpha_df.set_index(["date", "ticker"])["classical_alpha"], price_matrix
        )
        ml_ic = run_alphalens(
            ml_preds_df.set_index(["date", "ticker"])["ml_alpha"], price_matrix
        )

        feat.to_parquet(CACHE_DIR / "features.parquet")
        classical_alpha_df.to_parquet(CACHE_DIR / "classical_alpha.parquet")
        ml_preds_df.to_parquet(CACHE_DIR / "ml_alpha.parquet")
        factors.to_parquet(CACHE_DIR / "factors.parquet")
        (CACHE_DIR / "importances.json").write_text(json.dumps(importances))
        (CACHE_DIR / "classical_ic.json").write_text(json.dumps(classical_ic, default=str))
        (CACHE_DIR / "ml_ic.json").write_text(json.dumps(ml_ic, default=str))

        _state["status"] = "ready"
        return PipelineStatus(status="ready")
    except Exception as e:
        _state["status"] = "error"
        _state["error"] = f"{e}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        print(f"pipeline run took {time.time() - t0:.1f}s")


def _require_cache():
    if not (CACHE_DIR / "features.parquet").exists():
        raise HTTPException(status_code=409, detail="Pipeline hasn't been run yet. POST /pipeline/run first.")


@app.get("/pipeline/status")
async def pipeline_status():
    return _state


@app.get("/analysis/ic-decay")
async def ic_decay():
    _require_cache()
    classical_ic = json.loads((CACHE_DIR / "classical_ic.json").read_text())
    ml_ic = json.loads((CACHE_DIR / "ml_ic.json").read_text())
    return {"classical": classical_ic, "ml": ml_ic}


@app.get("/analysis/feature-importance")
async def feature_importance():
    _require_cache()
    return json.loads((CACHE_DIR / "importances.json").read_text())


@app.get("/universe/ranked")
async def ranked_universe(as_of: str | None = None, top_n: int = 50):
    """Combines classical + ML alpha into one ranked universe with attribution: how much of the
    combined score each stock's rank came from the classical factor model vs the ML layer.
    Both sub-scores are cross-sectionally z-scored each month before combining so one factor's
    scale can't dominate purely because of units."""
    _require_cache()
    classical_df = pd.read_parquet(CACHE_DIR / "classical_alpha.parquet")
    ml_df = pd.read_parquet(CACHE_DIR / "ml_alpha.parquet")

    merged = pd.merge(classical_df, ml_df, on=["date", "ticker"], how="inner")
    if as_of:
        target_date = pd.Timestamp(as_of)
        available = merged["date"].unique()
        nearest = min(available, key=lambda d: abs(pd.Timestamp(d) - target_date))
        merged = merged[merged["date"] == nearest]
    else:
        latest = merged["date"].max()
        merged = merged[merged["date"] == latest]

    if merged.empty:
        raise HTTPException(status_code=404, detail="No data for the requested date")

    def zscore(s):
        std = s.std()
        return (s - s.mean()) / std if std else s * 0

    merged["classical_z"] = zscore(merged["classical_alpha"])
    merged["ml_z"] = zscore(merged["ml_alpha"])
    merged["combined_score"] = 0.5 * merged["classical_z"] + 0.5 * merged["ml_z"]

    total_abs = (merged["classical_z"].abs() + merged["ml_z"].abs()).replace(0, 1)
    merged["classical_attribution_pct"] = (merged["classical_z"].abs() / total_abs * 100).round(1)
    merged["ml_attribution_pct"] = (merged["ml_z"].abs() / total_abs * 100).round(1)

    ranked = merged.sort_values("combined_score", ascending=False).head(top_n)
    ranked["rank"] = range(1, len(ranked) + 1)

    cols = ["rank", "ticker", "date", "combined_score", "classical_alpha", "ml_alpha",
            "classical_attribution_pct", "ml_attribution_pct"]
    result = ranked[cols].copy()
    result["date"] = result["date"].astype(str)
    return {"as_of": str(merged["date"].iloc[0]), "universe": result.to_dict(orient="records")}


@app.post("/data/upload")
async def upload_data(prices: UploadFile = File(...), fundamentals: UploadFile = File(...)):
    """Lets the frontend swap in real market_data.csv / fundamentals.csv without touching the
    server filesystem manually. Overwrites the files /pipeline/run reads from."""
    prices_bytes = await prices.read()
    fundamentals_bytes = await fundamentals.read()
    (DATA_DIR / "market_data.csv").write_bytes(prices_bytes)
    (DATA_DIR / "fundamentals.csv").write_bytes(fundamentals_bytes)
    return {"status": "uploaded", "note": "POST /pipeline/run to process the new data"}
