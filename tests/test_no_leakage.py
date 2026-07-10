"""Leak canaries against the real data (skipped until data/ is downloaded).

The synthetic tests prove pit_lookup's semantics; these two prove the shipped
data and the official forecast pairs keep the target out of reach end to end.
"""

import pandas as pd
import pytest

from src.config import DATA_DIR
# aliased: pytest would otherwise collect the imported test_index as a test
from src.data import load_signal
from src.data import test_index as _test_index
from src.data import train_index as _train_index
from src.features import PRICE, build_features
from src.interface import FeatureSpec

requires_data = pytest.mark.skipif(
    not (DATA_DIR / "target.parquet").exists(), reason="data/ not downloaded yet"
)


@requires_data
def test_same_interval_price_is_never_knowable():
    # A lag=0 price feature IS the target. forecast_index only keeps pairs
    # whose auction wasn't published at the origin, so on every official
    # train/test pair it must come back NaN — any non-NaN value is leakage.
    canary = FeatureSpec("price_day_ahead", PRICE, name="price_lag_0h")
    for index in (_train_index(), _test_index()):
        X = build_features([canary], index)
        assert X["price_lag_0h"].isna().all()


@requires_data
def test_price_snapshots_never_precede_auction_publication():
    # The as-of join is only as honest as available_at itself: no snapshot of
    # a price may exist in the feature panel before the auction published it.
    price = load_signal("price_day_ahead")
    first_snapshot = (
        price[price["zone_key"] == "DE"].groupby("target_time")["available_at"].min()
    )
    auction = pd.read_parquet(DATA_DIR / "target.parquet").set_index("target_time")[
        "available_at"
    ]
    auction = auction.reindex(first_snapshot.index).dropna()
    assert (first_snapshot.loc[auction.index] >= auction).all()
