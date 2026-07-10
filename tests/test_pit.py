"""Point-in-time lookup semantics: the heart of the case."""

from datetime import timedelta

import pandas as pd
from conftest import ts, write_signal

from src import data
from src.data import pit_lookup
from src.features import build_features
from src.interface import FeatureSpec


def pairs(*ref_target: tuple[str, str]) -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(
        [(ts(r), ts(t)) for r, t in ref_target], names=["ref_time", "target_time"]
    )


def test_picks_latest_snapshot_at_or_before_origin(data_dir):
    write_signal(
        data_dir,
        "price_day_ahead",
        [
            ("2026-05-10 12:00", "2026-05-08 13:00", 100.0),
            ("2026-05-10 12:00", "2026-05-09 00:00", 110.0),  # latest usable
            ("2026-05-10 12:00", "2026-05-09 13:00", 120.0),  # after origin: leak
        ],
    )
    points = pd.DatetimeIndex([ts("2026-05-10 12:00")])
    origins = pd.DatetimeIndex([ts("2026-05-09 00:00")])
    result = pit_lookup("price_day_ahead", ["value"], points, origins)
    assert result["value"].tolist() == [110.0]


def test_nothing_available_yields_nan(data_dir):
    # Settled grid-state only publishes after delivery: at the origin, the
    # lag=0 value must be NaN, not the future measurement.
    write_signal(
        data_dir,
        "zone_load_measurements",
        [("2026-05-10 12:00", "2026-05-10 14:06", 55.0)],
    )
    points = pd.DatetimeIndex([ts("2026-05-10 12:00")])
    origins = pd.DatetimeIndex([ts("2026-05-09 00:00")])
    result = pit_lookup("zone_load_measurements", ["value"], points, origins)
    assert result["value"].isna().all()


def test_lag_availability_depends_on_origin(data_dir):
    # The same 24h-lagged value is knowable from a short-lead origin but not
    # from a long-lead one: availability is a property of the pair.
    write_signal(
        data_dir,
        "price_day_ahead",
        [("2026-05-09 12:00", "2026-05-08 13:00", 80.0)],
    )
    index = pairs(
        ("2026-05-08 00:00", "2026-05-10 12:00"),  # lag point publishes 13:00, after this origin
        ("2026-05-09 00:00", "2026-05-10 12:00"),  # same lag point, already published here
    )
    X = build_features([FeatureSpec("price_day_ahead", "value", lag=timedelta(hours=24))], index)
    assert X.iloc[0, 0] != X.iloc[0, 0]  # NaN
    assert X.iloc[1, 0] == 80.0


def test_zone_key_filters_multi_zone_signals(data_dir):
    frame = pd.DataFrame(
        {
            "target_time": [ts("2026-05-10 12:00")] * 2,
            "zone_key": ["DE", "FR"],
            "available_at": [ts("2026-05-09 00:00")] * 2,
            "value": [100.0, 60.0],
        }
    )
    frame.to_parquet(data_dir / data.CATALOG["price_day_ahead"].filename)
    points = pd.DatetimeIndex([ts("2026-05-10 12:00")])
    origins = pd.DatetimeIndex([ts("2026-05-09 06:00")])
    de = pit_lookup("price_day_ahead", ["value"], points, origins)
    fr = pit_lookup("price_day_ahead", ["value"], points, origins, zone="FR")
    assert de["value"].tolist() == [100.0]
    assert fr["value"].tolist() == [60.0]


def test_hourly_weather_aligns_to_15min_grid(data_dir):
    write_signal(
        data_dir,
        "weather_ecmwf_production_solar",
        [("2026-05-10 12:00", "2026-05-09 00:00", 350.0)],
    )
    points = pd.DatetimeIndex([ts("2026-05-10 12:15"), ts("2026-05-10 12:45")])
    origins = pd.DatetimeIndex([ts("2026-05-09 06:00")] * 2)
    result = pit_lookup("weather_ecmwf_production_solar", ["value"], points, origins)
    assert result["value"].tolist() == [350.0, 350.0]
