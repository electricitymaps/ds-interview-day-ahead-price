"""The MODELS registry. Add your submission here.

A model = ModelSpec(features=[...], regressor=<anything with fit/predict>).
Run it with: uv run python -m src.run --model <name>
"""

from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import (
    PRICE,
    day_of_week,
    lead_time_h,
    price_fr_lag_24h,
    price_lag_168h,
    price_wow_change,
    ssrd_top_1,
)
from .interface import FeatureSpec, ModelSpec, Regressor


class WeeklyNaive(Regressor):
    """Predicts last week's price for the same interval. The bar to beat.

    The 168h lag is published well before every origin, so it works at all
    lead times — unlike shorter lags.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WeeklyNaive":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[price_lag_168h.output_name].to_numpy()


class RidgeReference(Regressor):
    """Ridge on median-imputed, standardized features.

    The imputer handles NaNs from lags whose reference interval is not yet
    published at the origin.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self._pipeline = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha)
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RidgeReference":
        self._pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._pipeline.predict(X))


MODELS: dict[str, ModelSpec] = {
    "baseline": ModelSpec(features=[price_lag_168h], regressor=WeeklyNaive()),
    "reference": ModelSpec(
        features=[
            FeatureSpec("price_day_ahead", PRICE, lag=timedelta(hours=48)),
            price_lag_168h,
            price_fr_lag_24h,
            ssrd_top_1,
            price_wow_change,
            day_of_week,
            lead_time_h,
        ],
        regressor=RidgeReference(),
    ),
}
