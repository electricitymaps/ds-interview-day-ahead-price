"""The contract your submission codes against.

A submission is a ModelSpec: a list of feature declarations plus a regressor.
Adding a feature = one declaration. Swapping a model = one registry entry in
models.py. The generic loop in run.py never changes.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from .config import ZONE


@runtime_checkable
class Regressor(Protocol):
    """What ModelSpec.regressor must provide: the sklearn fit/predict contract.

    This is a structural type — no inheritance needed. Any sklearn estimator,
    Pipeline, or LGBMRegressor already satisfies it; a hand-rolled class just
    needs these two methods (subclass Regressor if you want the type checker
    to hold you to the contract).
    """

    def fit(self, X: pd.DataFrame, y: pd.Series, /) -> Any: ...

    def predict(self, X: pd.DataFrame, /) -> Any: ...


@dataclass(frozen=True)
class FeatureSpec:
    """One column of X, taken from a signal at target_time - lag.

    The value used is the latest snapshot knowable at the forecast origin
    (available_at <= ref_time). lag=0 (the default) means "the value for
    target_time itself, as forecast/known at the origin". Whether a lagged
    value is knowable depends on the origin, so the same spec can be
    populated at short leads and NaN at long ones.
    """

    signal: str  # a key of data.CATALOG, e.g. "price_day_ahead"
    column: str  # a column of that signal's parquet
    lag: timedelta = timedelta(0)
    zone: str = ZONE  # price_day_ahead and zone_{load,production}_measurements carry other zones (see zone_key column)
    name: str | None = None  # X column name; defaults to "<column>[_<zone>]_lag_<h>h"

    @property
    def output_name(self) -> str:
        if self.name is not None:
            return self.name
        base = self.column if self.zone == ZONE else f"{self.column}_{self.zone}"
        hours = self.lag.total_seconds() / 3600
        return f"{base}_lag_{hours:g}h" if self.lag else base


@dataclass(frozen=True)
class DerivedFeature:
    """One column of X computed from other feature columns."""

    name: str
    inputs: tuple[FeatureSpec, ...]
    fn: Callable[..., pd.Series]  # receives one pd.Series per input, in order


@dataclass(frozen=True)
class CalendarFeature:
    """One column of X computed from the (ref_time, target_time) index alone.

    Always available (deterministic): calendar effects, lead time, etc.
    fn receives the MultiIndex; use index.get_level_values("target_time") or
    ("ref_time") as needed.
    """

    name: str
    fn: Callable[[pd.MultiIndex], pd.Series]


Feature = FeatureSpec | DerivedFeature | CalendarFeature


@dataclass
class ModelSpec:
    """A complete submission: what to feed the model, and the model itself.

    regressor is anything satisfying the Regressor protocol above: a plain
    sklearn estimator, a Pipeline, or your own class all work.
    """

    features: list[Feature]
    regressor: Regressor
