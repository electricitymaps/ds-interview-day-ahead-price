"""Data access: the signal catalog, loaders, and the point-in-time lookup.

Every signal parquet is a bitemporal panel:
    (target_time, zone_key, available_at, value columns...)
For one target_time there are several rows, one per snapshot, capturing how
the known value evolved. The point-in-time rule: a row is usable for a
forecast made at ref_time only if available_at <= ref_time.
"""

from dataclasses import dataclass
from functools import cache

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pandas.tseries.frequencies import to_offset

from .config import (
    DATA_DIR,
    MAX_LEAD_TIME,
    RESOLUTION,
    TEST_ORIGIN_END,
    TEST_ORIGIN_START,
    TEST_TARGET_END,
    TEST_TARGET_START,
    TRAIN_ORIGIN_END,
    TRAIN_ORIGIN_START,
    TRAIN_TARGET_END,
    TRAIN_TARGET_START,
    ZONE,
    origin_grid,
)


@dataclass(frozen=True)
class Signal:
    filename: str
    resolution: pd.Timedelta


CATALOG: dict[str, Signal] = {
    "price_day_ahead": Signal("price_day_ahead.parquet", pd.Timedelta(minutes=15)),
    "zone_load_measurements": Signal("zone_load_measurements.parquet", pd.Timedelta(minutes=15)),
    "zone_production_measurements": Signal(
        "zone_production_measurements.parquet", pd.Timedelta(minutes=15)
    ),
    "weather_ecmwf_load_total": Signal("weather_ecmwf_load_total.parquet", pd.Timedelta(hours=1)),
    "weather_ecmwf_production_solar": Signal(
        "weather_ecmwf_production_solar.parquet", pd.Timedelta(hours=1)
    ),
    "weather_ecmwf_production_wind": Signal(
        "weather_ecmwf_production_wind.parquet", pd.Timedelta(hours=1)
    ),
    "weather_noaa_load_total": Signal("weather_noaa_load_total.parquet", pd.Timedelta(hours=1)),
    "weather_noaa_production_solar": Signal(
        "weather_noaa_production_solar.parquet", pd.Timedelta(hours=1)
    ),
    "weather_noaa_production_wind": Signal(
        "weather_noaa_production_wind.parquet", pd.Timedelta(hours=1)
    ),
}


def catalog_columns(signal: str) -> list[str]:
    """Value columns of a signal, without loading the data. See DATA_DICTIONARY.md."""
    schema = pq.read_schema(DATA_DIR / CATALOG[signal].filename)
    return [c for c in schema.names if c not in ("target_time", "zone_key", "available_at")]


@cache
def load_signal(signal: str) -> pd.DataFrame:
    """The full panel of one signal — handy for exploration."""
    return pd.read_parquet(DATA_DIR / CATALOG[signal].filename)


@cache
def _panel_slice(signal: str, zone: str, columns: tuple[str, ...]) -> pd.DataFrame:
    """Zone-filtered, column-projected slice of a signal panel (cached).

    Reads only the requested columns (parquet is columnar) so lookups never
    pay for the ~180-column weather panels they don't use.
    """
    panel = pd.read_parquet(
        DATA_DIR / CATALOG[signal].filename,
        columns=["target_time", "zone_key", "available_at", *columns],
    )
    return panel[panel["zone_key"] == zone].drop(columns="zone_key")


def pit_lookup(
    signal: str,
    columns: list[str],
    points: pd.DatetimeIndex,
    origins: pd.DatetimeIndex,
    zone: str = ZONE,
) -> pd.DataFrame:
    """Point-in-time values of `columns` at times `points`, as knowable at `origins`.

    For each (point, origin) pair, returns the latest snapshot of the signal at
    target_time == point with available_at <= origin; NaN if nothing was
    available yet. points and origins are aligned arrays (one row of X each).
    """
    resolution = CATALOG[signal].resolution
    request = pd.DataFrame({"point": points.floor(to_offset(resolution)), "origin": origins})
    # Many rows share a (point, origin) pair (e.g. 4 15-min targets per weather
    # hour): resolve each pair once, then map back onto the full request.
    unique = request.drop_duplicates()
    panel = _panel_slice(signal, zone, tuple(columns))
    merged = unique.merge(panel, left_on="point", right_on="target_time", how="left")
    merged = merged[merged["available_at"] <= merged["origin"]]
    latest = merged.sort_values("available_at").groupby(["point", "origin"]).tail(1)
    resolved = request.merge(latest, on=["point", "origin"], how="left")
    return resolved[columns]


@cache
def _target_frame() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "target.parquet")


def load_target() -> pd.Series:
    """Realized day-ahead price per target_time (EUR/MWh)."""
    return _target_frame().set_index("target_time")["target_value"].sort_index()


def forecast_index(
    origin_start: pd.Timestamp,
    origin_end: pd.Timestamp,
    target_start: pd.Timestamp,
    target_end: pd.Timestamp,
) -> pd.MultiIndex:
    """The (ref_time, target_time) pairs to forecast.

    For each 6-hourly origin, every 15-min interval up to 72h ahead — except
    targets whose auction result was already published at the origin
    (ref_time < available_at, mirroring gridcast's target alignment). You never
    forecast, or get credit for, a price that was already known.
    """
    origins = origin_grid(origin_start, origin_end)
    leads = pd.timedelta_range(RESOLUTION, MAX_LEAD_TIME, freq=RESOLUTION)
    ref_times = np.repeat(origins, len(leads))
    pairs = pd.DataFrame(
        {"ref_time": ref_times, "target_time": ref_times + np.tile(leads, len(origins))}
    )
    pairs = pairs[pairs["target_time"].between(target_start, target_end)]
    published = _target_frame()[["target_time", "available_at"]]
    pairs = pairs.merge(published, on="target_time", how="inner")
    pairs = pairs[pairs["ref_time"] < pairs["available_at"]]
    return pd.MultiIndex.from_frame(
        pairs[["ref_time", "target_time"]].sort_values(["ref_time", "target_time"])
    )


def train_index() -> pd.MultiIndex:
    """The official training pairs."""
    return forecast_index(
        TRAIN_ORIGIN_START, TRAIN_ORIGIN_END, TRAIN_TARGET_START, TRAIN_TARGET_END
    )


def test_index() -> pd.MultiIndex:
    """The official test pairs — exactly what predictions.parquet must cover."""
    return forecast_index(TEST_ORIGIN_START, TEST_ORIGIN_END, TEST_TARGET_START, TEST_TARGET_END)


def target_on(index: pd.MultiIndex) -> pd.Series:
    """Realized target values aligned onto a (ref_time, target_time) index."""
    values = load_target().reindex(index.get_level_values("target_time"))
    return pd.Series(values.to_numpy(), index=index, name="target_value")
