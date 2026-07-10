from pathlib import Path
from typing import cast

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

ZONE = "DE"  # the home zone: the target, and the default zone of every feature
RESOLUTION = pd.Timedelta(minutes=15)
ORIGIN_RESOLUTION = pd.Timedelta(hours=6)
MAX_LEAD_TIME = pd.Timedelta(hours=72)

# Training: origins spanning the train window; targets reach up to 72h beyond each
# origin (Apr 27 18:00 + 72h = Apr 30 18:00, safely before the May test targets).
TRAIN_ORIGIN_START = pd.Timestamp("2025-10-01 00:00", tz="UTC")
TRAIN_ORIGIN_END = pd.Timestamp("2026-04-27 18:00", tz="UTC")
TRAIN_TARGET_START = TRAIN_ORIGIN_START
TRAIN_TARGET_END = pd.Timestamp("2026-04-30 23:45", tz="UTC")

# Test: origins every 6h from Apr 29 to May 29; only May targets are scored.
TEST_ORIGIN_START = pd.Timestamp("2026-04-29 00:00", tz="UTC")
TEST_ORIGIN_END = pd.Timestamp("2026-05-29 18:00", tz="UTC")
TEST_TARGET_START = pd.Timestamp("2026-05-01 00:00", tz="UTC")
TEST_TARGET_END = pd.Timestamp("2026-05-31 23:45", tz="UTC")


def origin_grid(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Forecast origins (ref_times): every 6 hours, 00:00/06:00/12:00/18:00 UTC."""
    return pd.date_range(start, end, freq=ORIGIN_RESOLUTION, name="ref_time")


def datetime_level(index: pd.MultiIndex, name: str) -> pd.DatetimeIndex:
    """A tz-aware datetime level of the index, typed as DatetimeIndex.

    `MultiIndex.get_level_values` is stubbed to return the base `Index`, which
    drops the datetime API (`.dayofweek`, timedelta subtraction, ...). Our
    `ref_time`/`target_time` levels are always datetime, so narrow the type.
    """
    return cast(pd.DatetimeIndex, index.get_level_values(name))


def lead_time_hours(index: pd.MultiIndex) -> pd.Series:
    """Lead time in hours for a (ref_time, target_time) forecast index."""
    delta = datetime_level(index, "target_time") - datetime_level(index, "ref_time")
    return pd.Series(delta.total_seconds() / 3600, index=index)
