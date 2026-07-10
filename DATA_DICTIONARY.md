# Data dictionary

All files live in `data/`. Every signal shares the same bitemporal shape:

| column         | meaning                                                      |
| -------------- | ------------------------------------------------------------ |
| `target_time`  | the delivery interval the value describes (tz-aware UTC)     |
| `zone_key`     | which zone the row describes (price and grid-state files are multi-zone: DE + 11 neighbours; target and weather are `DE` only) |
| `available_at` | when that value became known / was published (tz-aware UTC)  |
| value columns  | signal-specific, listed below                                |

For one `target_time` there are several rows, one per `available_at` snapshot. Use
`src.data.catalog_columns(<signal>)` to list any signal's columns from code.

## Target

**`target.parquet`** — `(target_time, zone_key, target_value, available_at)`. The
realized day-ahead auction price (`zone_key` = `DE`), EUR/MWh, 15-min. `available_at` is the auction
publication time (~13:00 CET on D-1, one batch per delivery day).

## Market

**`price_day_ahead.parquet`** — 15-min, **multi-zone**: `zone_key`
carries DE plus its 11 electrical neighbours
(AT, BE, CH, CZ, DK-DK1, DK-DK2, FR, NL, NO-NO2, PL, SE-SE4).
One value column: `time_aligned_day_ahead_price_price` (EUR/MWh in every zone,
including the non-euro ones). The auction price for delivery day D appears only in
snapshots after ~13:00 CET on D-1 — and coupled zones clear in the same auction, so
neighbour prices for the delivery day are exactly as unknown as DE's own.

## Grid state (settled = consolidated measurements)

These are the **consolidated** (final) grid measurements, **multi-zone** like the
price file: `zone_key` carries DE plus the same 11 neighbours — except production,
where CH is absent (Switzerland publishes no comparable production measurements, so
CH has load only). Despite the bitemporal framing above, each `(zone_key,
target_time)` carries **exactly one** `available_at` (verified for every zone) —
there is no revision trail to reconcile. Publication lag is a **fixed offset**, not
the variable ~2–3h latency one might assume: **load = `target_time` + 2h06m**,
**production = `target_time` + 3h10m**, constant across every interval and every
zone. So the point-in-time rule is arithmetic: a grid interval `T` is usable at any
origin `ref_time ≥ T + offset`.

**`zone_load_measurements.parquet`** — 15-min, 12 zones.
One value column: `time_aligned_load_total` (MW).

**`zone_production_measurements.parquet`** — 15-min, 11 zones (no CH). Production per mode (MW):
`time_aligned_production_breakdown_{biomass,coal,gas,geothermal,hydro,nuclear,oil,solar,wind,unknown}`,
storage flows `..._{battery_storage,hydro_storage}` (MW; positive = charging,
negative = discharging), and shares
`..._power_production_percent_{fossil,renewable}` (%).
Modes a zone does not have are all-NaN for that zone (e.g. `nuclear` in DE, AT and
PL; `hydro` in NL and DK-DK2; `battery_storage` in most zones) — check before using
a column outside DE. Coverage gaps: DK-DK2 production is missing ~16% of intervals
(22,837 of 27,165 target times) and DK-DK1 ~1%; every other zone is complete.

## Weather (forecasts — the delivery day IS knowable at the origin)

1-hour `target_time`, 4 issuances per day (`available_at` every ~6h), horizon ~5–6
days. Columns follow `<variable>_top_<i>` with **exactly i = 1..10 for every
variable**: the i-th best-selected grid cell for that variable and purpose.
**`top_i` is a per-variable rank — `x_top_1` and `y_top_1` are generally NOT the same
location.** Don't average across cells blindly; they are deliberately diverse
localizations. Exact column names: `src.data.catalog_columns(<signal>)`.

| file | variables (× top_1..top_N) |
| ---- | -------------------------- |
| `weather_ecmwf_production_solar.parquet` | `ssrd`, `strd`, `tmp_2m` |
| `weather_ecmwf_production_wind.parquet` | `wind_{speed,dir,u_norm,v_norm}_{10m,100m,500hpa,850hpa}`, `pressure_0m`, `tmp_2m` |
| `weather_ecmwf_load_total.parquet` | `tmp_2m`, `dewpoint_tmp_2m`, `tmp_850hpa`, `pressure_0m`, `gph_{500,850}hpa`, `rel_humidity_{600,850}hpa`, `spec_humidity_{600,700}hpa`, `tcwv`, `snow_albedo`, `wind_{speed,dir,u_norm,v_norm}_10m` |
| `weather_noaa_production_solar.parquet` | `dswrf_avg`, `uswrf_avg`, `{l,m,h}cdc_{avg,instant}`, `tcdc_instant` (no `tcdc_avg`), `sunshine_duration`, `tmp_2m` |
| `weather_noaa_production_wind.parquet` | `wind_{speed,dir,u_norm,v_norm}_{10m,80m,100m}`, `wind_speed_gust_0m`, `pressure_{0m,80m}`, `tmp_2m` |
| `weather_noaa_load_total.parquet` | `tmp_{0m,2m}`, `rel_humidity_2m`, `pressure_0m`, `cape`, `wind_{speed,dir,u_norm,v_norm}_10m` |

Units: temperatures in °C, pressure in Pa, wind speed in m/s, wind direction in
degrees, `u_norm`/`v_norm` are unit-vector wind components, radiation (`ssrd`,
`strd`, `dswrf_avg`, `uswrf_avg`) in W/m² (mean flux — ECMWF's native accumulated
J/m² is already de-accumulated), cloud cover (`*cdc*`) in %,
`tcwv` in kg/m², `cape` in J/kg, `gph_*` in m.
