# Day-ahead electricity price forecasting — take-home case

You are forecasting **15-minute day-ahead electricity prices for Germany (DE)**.

Every feature parquet carries **`target_time`** (the delivery interval a value
describes) and **`available_at`** (when that value became known), so you can infer what
information is available at any forecast time. Forecasts are made from rolling
**origins (`ref_time`) every 6 hours** (00/06/12/18 UTC): from each origin, predict
every 15-min interval up to **72 hours ahead**, except intervals whose auction result
was already published at that origin (`src.data.forecast_index` builds exactly this
set of `(ref_time, target_time)` pairs). What is knowable differs per origin — the
same lagged feature can be published for one origin and not yet auctioned for another.

You have ~4 hours. You will not have time to do everything one could imagine here.
**Ship something that runs end to end, prioritise deliberately, and tell us what you
chose to skip and why.**

## Setup (~5 minutes)

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
curl -L -o data.tar.gz "<signed-url-you-received>"
tar xzf data.tar.gz          # unpacks into data/
uv run pytest                # sanity check
uv run python -m src.run --model reference   # first predictions end to end
```

## The data

`data/` contains nine feature signals plus the target — see `DATA_DICTIONARY.md` for
every column. Each signal is a bitemporal panel: for one `target_time` there are
several rows, one per `available_at` snapshot, capturing how the known value evolved.
Prices and grid measurements (load, production) are **multi-zone**: `zone_key` carries
DE plus its 11 electrical neighbours (CH has load but no production data); weather is
DE-only. Select a neighbour via the `zone` argument of `FeatureSpec` / `pit_lookup`.

- **Train:** origins 1 Oct 2025 – 27 Apr 2026 (targets through 30 Apr).
- **Test:** origins 29 Apr – 29 May 2026 (6-hourly); **only May targets are scored**,
  with per-lead-band metrics (0–24h / 24–48h / 48–72h).
- The test-month target ships with the data so you can self-score. We recompute your
  score from your predictions and re-run your code; do not fit or tune on the test
  month. `run.py` prints the test score for convenience — for model iteration, carve
  a validation window out of the training period instead (your WRITEUP asks how).

## The scaffold

The plumbing is done so you can spend your time on features and models:

- `src/interface.py` — `FeatureSpec` / `DerivedFeature` / `CalendarFeature` / `ModelSpec`.
- `src/data.py` — signal catalog (`CATALOG`, `catalog_columns()`), loaders, and
  `pit_lookup`, the point-in-time primitive.
- `src/features.py` — `build_features()` plus one worked example per mechanism
  (price lag, weather forecast, derived feature, calendar).
- `src/models.py` — the `MODELS` registry: `baseline` (weekly naive — the bar to
  beat) and `reference` (minimal Ridge wiring demo).
- `src/run.py` / `src/score.py` — the generic loop and the scorer.

Adding a feature is one declaration; adding a model is one registry entry:

```python
MODELS["mine"] = ModelSpec(
    features=[price_lag_168h, ssrd_top_1, my_new_feature, ...],
    regressor=LGBMRegressor(),   # anything with fit/predict
)
```

Declare features you reuse in `src/features.py` (one-off specs can sit inline in the
feature list); register models in `src/models.py`, or grow into new modules under
`src/` / `solution/` — just make sure `uv run python -m src.run --model <yours>`
reproduces your submission.

## Deliverables

Send the whole repo back (repo link or zip — not just the parquet). Hand-in checklist:

- [ ] **`<model>_predictions.parquet`** from your **final** model (`run.py` names the
      file after the registry entry). Columns: `ref_time`, `target_time` (exactly the
      pairs from `src.data.test_index()` — 23,048 with the shipped windows),
      `price_forecast` (no NaNs), and `model`.
- [ ] **It validates**: `uv run python -m src.score <model>_predictions.parquet`
      exits clean and prints your scores.
- [ ] **`WRITEUP.md`** with every section filled in. Section 4 (what you deprioritised
      and why) matters as much as your score. If you ran several models, say which
      file is your submission.
- [ ] **Your code**, with your final model registered in `MODELS`, such that a fresh
      clone + data download + `uv run python -m src.run --model <yours>` regenerates
      your parquet. We will re-run it — predictions we can't reproduce don't count.
- [ ] **`uv run pytest` still passes.**

## Ground rules

- **AI tools are allowed** — you would use them on the job. Mention notable usage in
  the writeup. A follow-up debrief will go through your choices in depth.
- Beat the weekly-naive baseline on the test month; beyond that, we grade reasoning,
  correctness, and prioritisation over raw score.
