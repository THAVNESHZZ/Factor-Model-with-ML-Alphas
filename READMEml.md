# Factor Model with ML Alphas

Fama-French style baseline factor model, augmented with a LightGBM ML alpha layer, validated with
Alphalens (IC, turnover, decay curves). FastAPI backend, Next.js 14 frontend.

## Architecture

```
backend/
  app/
    ingestion.py     # point-in-time CSV ingestion (no lookahead)
    features.py      # ratios, 12-1 momentum, rolling vol/Sharpe, short-interest stub, fwd returns
    factor_model.py   # constructs Market/SMB/HML from the universe, rolling classical-alpha regression
    ml_alpha.py       # LightGBM, expanding walk-forward validation
    analysis.py       # Alphalens wrapper: IC, turnover, decay
    main.py           # FastAPI app: /pipeline/run, /universe/ranked, /analysis/*, /data/upload
  data/
    generate_synthetic_data.py   # generates market_data.csv + fundamentals.csv for testing
  requirements.txt
  Dockerfile
  railway.json
frontend/
  app/page.tsx                    # dashboard: run pipeline, ranked universe, IC/attribution charts
  components/                     # PipelineControl, RankedUniverseTable, AnalysisCharts
  lib/api.ts                      # typed API client
```

## Running locally

**Backend**
```
cd backend
pip install -r requirements.txt
python data/generate_synthetic_data.py   # only if you don't have your own market_data.csv/fundamentals.csv yet
uvicorn app.main:app --reload --port 8000
```
Then `POST http://localhost:8000/pipeline/run` (or click "Run pipeline" in the frontend) before
hitting any of the `/universe` or `/analysis` endpoints — they read from a cache the pipeline run
populates, and return 409 until it's been run at least once.

**Frontend**
```
cd frontend
npm install
npm run dev
```
Set `NEXT_PUBLIC_API_URL` in `.env.local` to point at your backend (defaults to `localhost:8000`).

## Using your own data

Your CSV schema (as agreed):
- `market_data.csv`: `date, ticker, open, high, low, close, adj_close, volume` — daily
- `fundamentals.csv`: `date, ticker, book_value, net_income, total_equity, total_debt` — quarterly,
  `date` = the public **release** date, not quarter-end (this is what keeps the merge lookahead-free)

Either drop the files into `backend/data/` directly, or `POST /data/upload` (multipart, fields
`prices` and `fundamentals`) from the running frontend/API — same file, no lookahead assumptions
changed.

## Deploying

- **Backend** → Render or Railway, using `backend/Dockerfile`. `railway.json` is included; on Render,
  just point it at the Dockerfile, no extra config needed. Set `PORT` if your platform requires it
  (Dockerfile already reads `$PORT`).
- **Frontend** → Vercel, zero-config for Next.js (just import the `frontend/` directory as the
  project root). Set `NEXT_PUBLIC_API_URL` to your deployed backend URL in Vercel's env vars.

## Bugs found and fixed in the uploaded fragments

1. **`Dockerfile` CMD pointed at the wrong module.** It ran `uvicorn main:app`, but the actual
   FastAPI instance lives at `app/main.py` inside a package — fixed to `uvicorn app.main:app`.
   This would have failed immediately on container start (`ModuleNotFoundError`).
2. **Missing `libgomp1` in the Dockerfile.** LightGBM's compiled backend needs the OpenMP runtime,
   which isn't in `python:3.11-slim` by default — without it, `import lightgbm` throws
   `libgomp.so.1: cannot open shared object file` at runtime, not build time, so it's an easy one
   to miss until deploy. Added to the `apt-get install` line.
3. **CORS bug in the stub `main.py`:** `allow_origins=["*"]` combined with `allow_credentials=True`
   is invalid per the CORS spec — browsers silently block the credentialed cross-origin request,
   so the deployed frontend would have looked broken with no obvious server-side error. Fixed by
   dropping `allow_credentials` (this API uses no cookies/session auth, so it isn't needed).
4. **`vercel.json` targeted the wrong framework.** It set `"framework": "vite"` with an SPA
   catch-all rewrite to `index.html` — but the frontend is Next.js, which has its own router and
   serverless functions; a Vite-style SPA rewrite would break Next.js routing entirely. Next.js on
   Vercel needs no `vercel.json` at all (zero-config), so it's simply not included in this build.
5. **Schema/data mismatch:** the Gemini-generated synthetic panel script produced a single monthly
   parquet file with pre-mixed ratios, no release-date semantics, and no raw price series — it
   doesn't fit the point-in-time ingestion pipeline the CSV schema requires. Replaced with a new
   generator (`backend/data/generate_synthetic_data.py`) matching the agreed schema exactly.

## Known limitations (real, not hidden)

- **B/M ratio uses `adj_close` as a market-value proxy**, not true market cap, because the agreed
  fundamentals schema has no `shares_outstanding` column. Fine for cross-sectional ranking in this
  demo; before using real data, add `shares_outstanding` (or source market cap directly from your
  price vendor) and update `features.py::compute_ratios`.
- **Short interest is a stub** (uncorrelated placeholder noise), per your explicit decision — swap
  in a real point-in-time short-interest feed by replacing `add_short_interest_stub` in
  `features.py`; the rest of the pipeline (feature list, model, Alphalens) doesn't need to change.
- **The ML alpha's out-of-sample IC on the synthetic data is weak/mixed** (near zero, sometimes
  negative at longer horizons). This is an honest result, not a bug: the synthetic generator embeds
  a deliberately small, noisy true signal (like real equity factors), and walk-forward validation
  won't fabricate predictability that isn't there. Real fundamentals/prices will likely show a
  different (and worth investigating) IC profile.
- **Frontend is functional-first, not yet the 3D/parallax experience** — per your stated priority.
  The Next.js scaffold you got from o3 (NavBar, HeroParallax, framer-motion) is a good phase-2
  starting point once the data story is solid; happy to build that pass next.
