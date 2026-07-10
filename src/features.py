"""Generic feature builder + one worked example per mechanism.

X has one row per (ref_time, target_time) pair: the same delivery interval is
forecast from several origins, and what is knowable differs per origin.
"""

from datetime import timedelta

import pandas as pd

from .config import datetime_level, lead_time_hours
from .data import pit_lookup
from .interface import CalendarFeature, DerivedFeature, Feature, FeatureSpec

PRICE = "time_aligned_day_ahead_price_price"


def _resolve_specs(specs: list[FeatureSpec], index: pd.MultiIndex) -> dict[FeatureSpec, pd.Series]:
    """One pit_lookup per (signal, zone, lag) — extra columns ride the same join,
    so declaring many columns from one signal costs almost nothing."""
    target_times = datetime_level(index, "target_time")
    origins = datetime_level(index, "ref_time")
    groups: dict[tuple, list[FeatureSpec]] = {}
    for spec in dict.fromkeys(specs):
        groups.setdefault((spec.signal, spec.zone, spec.lag), []).append(spec)
    resolved: dict[FeatureSpec, pd.Series] = {}
    for (signal, zone, lag), group in groups.items():
        columns = list(dict.fromkeys(spec.column for spec in group))
        values = pit_lookup(signal, columns, target_times - lag, origins, zone=zone)
        for spec in group:
            resolved[spec] = pd.Series(values[spec.column].to_numpy(), index=index)
    return resolved


def build_features(features: list[Feature], index: pd.MultiIndex) -> pd.DataFrame:
    """Materialise X for a (ref_time, target_time) index, point-in-time correct."""
    specs = [f for f in features if isinstance(f, FeatureSpec)]
    specs += [s for f in features if isinstance(f, DerivedFeature) for s in f.inputs]
    resolved = _resolve_specs(specs, index)

    columns: dict[str, pd.Series] = {}
    for feature in features:
        match feature:
            case FeatureSpec():
                columns[feature.output_name] = resolved[feature]
            case DerivedFeature():
                columns[feature.name] = feature.fn(*(resolved[s] for s in feature.inputs))
            case CalendarFeature():
                columns[feature.name] = pd.Series(feature.fn(index), index=index)
    return pd.DataFrame(columns, index=index)


# --- Worked examples (one per mechanism; generalise from these) ---------------

# Autoregressive lag: own-zone price one week earlier: published well before every
# origin, so populated at all lead times — unlike shorter lags. The weekly-naive
# dummy_model predicts with it.
price_lag_168h = FeatureSpec(
    "price_day_ahead", PRICE, lag=timedelta(hours=168), name="price_lag_168h"
)

# Neighbour-zone price (market coupling). Coupled zones clear in the same auction,
# so neighbour prices are exactly as unknown as our own at any given origin.
price_fr_lag_24h = FeatureSpec("price_day_ahead", PRICE, lag=timedelta(hours=24), zone="FR")

# Weather is a forecast: the delivery-time value IS knowable at the origin (lag=0).
ssrd_top_1 = FeatureSpec("weather_ecmwf_production_solar", "ssrd_top_1")

# Derived feature: any fn of any FeatureSpec inputs — mix signals, zones, or
# lags freely. This one is deliberately plain wiring: yesterday's price vs the
# same interval a week before (NaNs in any input propagate to the output).
price_wow_change = DerivedFeature(
    name="price_wow_change",
    inputs=(
        FeatureSpec("price_day_ahead", PRICE, lag=timedelta(hours=24)),
        FeatureSpec("price_day_ahead", PRICE, lag=timedelta(hours=24 + 168)),
    ),
    fn=lambda yesterday, week_before: yesterday - week_before,
)

# Deterministic features of the index: calendar, and the forecast lead time.
day_of_week = CalendarFeature(
    "day_of_week",
    lambda index: pd.Series(datetime_level(index, "target_time").dayofweek, index=index),
)
lead_time_h = CalendarFeature("lead_time_h", lead_time_hours)
