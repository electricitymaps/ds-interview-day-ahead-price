"""forecast_index semantics + the generic loop end to end on synthetic data."""

import numpy as np
import pandas as pd
import pytest
from conftest import ts

from src import data
from src.data import forecast_index, target_on
from src.features import build_features, day_of_week
from src.interface import FeatureSpec, ModelSpec
from src.models import WeeklyNaive
from src.score import evaluate


@pytest.fixture
def synthetic_data_dir(data_dir):
    index = pd.date_range("2026-05-01", "2026-05-21 23:45", freq="15min", tz="UTC")
    price = 100 + 20 * np.sin(np.arange(len(index)) * 2 * np.pi / 96)
    published = index.floor("D") - pd.Timedelta(hours=13)  # 11:00 UTC on D-1
    pd.DataFrame(
        {"target_time": index, "zone_key": "DE", "available_at": published, "value": price}
    ).to_parquet(data_dir / data.CATALOG["price_day_ahead"].filename)
    pd.DataFrame(
        {"target_time": index, "target_value": price, "available_at": published}
    ).to_parquet(data_dir / "target.parquet")
    return data_dir


def test_forecast_index_respects_publication(synthetic_data_dir):
    index = forecast_index(
        ts("2026-05-10 00:00"), ts("2026-05-10 18:00"), ts("2026-05-01"), ts("2026-05-21 23:45")
    )
    frame = index.to_frame(index=False)

    # Max lead is 72h; leads are strictly positive.
    lead = frame["target_time"] - frame["ref_time"]
    assert lead.max() <= pd.Timedelta(hours=72) and lead.min() > pd.Timedelta(0)

    # From the 00:00 origin, May 10 is already published (May 9 11:00) -> the
    # first forecastable delivery day is May 11.
    from_midnight = frame[frame["ref_time"] == ts("2026-05-10 00:00")]
    assert from_midnight["target_time"].min() == ts("2026-05-11 00:00")

    # From the 12:00 origin, May 11 was published at 11:00 -> first day is May 12.
    from_noon = frame[frame["ref_time"] == ts("2026-05-10 12:00")]
    assert from_noon["target_time"].min() == ts("2026-05-12 00:00")


def test_fit_predict_score(synthetic_data_dir):
    spec = ModelSpec(
        features=[
            FeatureSpec("price_day_ahead", "value", lag=pd.Timedelta(hours=168), name="price_lag_168h"),
            day_of_week,
        ],
        regressor=WeeklyNaive(),
    )
    train_index = forecast_index(
        ts("2026-05-08"), ts("2026-05-12 18:00"), ts("2026-05-01"), ts("2026-05-15 23:45")
    )
    test_index = forecast_index(
        ts("2026-05-13"), ts("2026-05-17 18:00"), ts("2026-05-16"), ts("2026-05-20 23:45")
    )

    X_train = build_features(spec.features, train_index)
    spec.regressor.fit(X_train, target_on(train_index))
    X_test = build_features(spec.features, test_index)
    predictions = pd.Series(spec.regressor.predict(X_test), index=test_index)

    # The synthetic price has a weekly-periodic daily cycle: weekly naive is exact.
    metrics = evaluate(target_on(test_index), predictions)
    assert metrics["nmae"] == pytest.approx(0.0, abs=1e-9)


def test_forecast_index_drops_early_published_targets(synthetic_data_dir):
    frame = pd.read_parquet(synthetic_data_dir / "target.parquet")
    # Pretend one delivery day was published four days early.
    early = frame["target_time"].dt.date == pd.Timestamp("2026-05-10").date()
    frame.loc[early, "available_at"] -= pd.Timedelta(days=4)
    frame.to_parquet(synthetic_data_dir / "target.parquet")
    data._target_frame.cache_clear()

    index = forecast_index(
        ts("2026-05-08 00:00"), ts("2026-05-08 00:00"), ts("2026-05-01"), ts("2026-05-21 23:45")
    )
    days = index.get_level_values("target_time").date
    assert not (days == pd.Timestamp("2026-05-10").date()).any()
    # May 9 is fully forecastable; the 72h horizon just reaches May 11 00:00.
    assert (days == pd.Timestamp("2026-05-09").date()).sum() == 96
    assert (days == pd.Timestamp("2026-05-11").date()).sum() == 1
