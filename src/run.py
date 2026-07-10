"""End-to-end run: build X/y on (ref_time, target_time) pairs -> fit ->
predict the test window from every test origin -> predictions.parquet."""

import argparse

import pandas as pd

from .data import target_on, test_index, train_index
from .features import build_features
from .models import MODELS
from .score import evaluate, format_metrics


def regressor_name(regressor) -> str:
    """'Pipeline' says nothing — name the steps (e.g. SimpleImputer -> Ridge)."""
    steps = getattr(regressor, "steps", None)
    if steps:
        return " -> ".join(type(estimator).__name__ for _, estimator in steps)
    return type(regressor).__name__


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="baseline", choices=sorted(MODELS))
    parser.add_argument("--output", default=None, help="defaults to <model>_predictions.parquet")
    args = parser.parse_args()
    spec = MODELS[args.model]
    output = args.output or f"{args.model}_predictions.parquet"

    train = train_index()
    print(f"[{args.model}] building features for {len(train):,} training pairs...")
    X_train = build_features(spec.features, train)
    y_train = target_on(train).dropna()
    X_train = X_train.loc[y_train.index]

    print(f"[{args.model}] fitting {regressor_name(spec.regressor)}...")
    spec.regressor.fit(X_train, y_train)

    test = test_index()
    X_test = build_features(spec.features, test)
    predictions = pd.Series(spec.regressor.predict(X_test), index=test, name="price_forecast")
    frame = predictions.reset_index()  # plain columns: ref_time, target_time, price_forecast
    frame["model"] = args.model
    frame.to_parquet(output, index=False)
    print(f"[{args.model}] wrote {len(frame):,} predictions to {output}")

    y_test = target_on(test).dropna()
    metrics = evaluate(y_test, predictions.loc[y_test.index])
    print(f"[{args.model}] test window: {format_metrics(metrics)}")


if __name__ == "__main__":
    main()
