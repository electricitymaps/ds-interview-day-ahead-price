"""Score a predictions.parquet against the shipped test target.

Usage: uv run python -m src.score predictions.parquet
"""

import argparse
from typing import cast

import numpy as np
import pandas as pd

from .config import MAX_LEAD_TIME, lead_time_hours
from .data import target_on, test_index


def evaluate(y: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    error = y_pred - y
    return {
        "nmae": float(error.abs().mean() / y.mean()),
        "rmse": float(np.sqrt((error**2).mean())),
    }


def format_metrics(metrics: dict[str, float]) -> str:
    return ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", help="path to predictions.parquet")
    args = parser.parse_args()

    frame = pd.read_parquet(args.predictions)
    expected = test_index()

    problems = []
    missing = [c for c in ("ref_time", "target_time", "price_forecast") if c not in frame.columns]
    if missing:
        problems.append(f"missing columns: {', '.join(missing)}")
    else:
        if frame["price_forecast"].isna().any():
            problems.append("price_forecast contains NaNs")
        pairs = pd.MultiIndex.from_frame(frame[["ref_time", "target_time"]])
        if not pairs.sort_values().equals(expected):
            problems.append(
                f"(ref_time, target_time) must be exactly the {len(expected):,} pairs "
                f"of the test window (got {len(frame):,} rows) — build them with "
                "src.data.test_index"
            )
    if problems:
        raise SystemExit("invalid submission: " + "; ".join(problems))

    if "model" in frame.columns:
        print(f"model: {', '.join(frame['model'].unique())}")
    predictions = frame.set_index(["ref_time", "target_time"])
    y = target_on(expected).dropna()
    y_pred = predictions["price_forecast"].loc[y.index]
    print(f"overall: {format_metrics(evaluate(y, y_pred))}")

    edges = range(0, int(MAX_LEAD_TIME / pd.Timedelta(hours=1)) + 1, 24)
    bands = pd.cut(
        lead_time_hours(cast(pd.MultiIndex, y.index)),
        bins=list(edges),
        labels=[f"{a}-{b}h" for a, b in zip(edges, list(edges)[1:], strict=False)],
    )
    for band, y_band in y.groupby(bands, observed=True):
        print(f"lead {band}: {format_metrics(evaluate(y_band, y_pred.loc[y_band.index]))}")


if __name__ == "__main__":
    main()
