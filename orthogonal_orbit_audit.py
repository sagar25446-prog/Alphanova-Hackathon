"""Local audit for the Orthogonal Orbit candidate; not a submission."""

import ast
import time

import numpy as np
import pandas as pd

from alphanova_orthogonal_orbit import OrthogonalOrbitPredictor
from data_loader import list_periods, load_period
from evaluation import backtest, check_cross_sectional_demean


def main() -> None:
    with open("alphanova_orthogonal_orbit.py", encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    print("top_level=" + ",".join(type(node).__name__ for node in tree.body))

    pnl_parts = []
    gross_parts = []
    turnover_parts = []
    fold_scores = []
    train_times = []
    predict_times = []
    for period in list_periods():
        data = load_period(period)
        n_val = min(800, int(0.2 * len(data["features"])))
        n_train = len(data["features"]) - n_val
        predictor = OrthogonalOrbitPredictor()
        started = time.perf_counter()
        predictor.train(data["features"].iloc[:n_train], data["target"].iloc[:n_train])
        train_times.append(time.perf_counter() - started)
        started = time.perf_counter()
        prediction = predictor.predict(data["features"].iloc[n_train:])
        predict_times.append(time.perf_counter() - started)
        returns = data["returns"].iloc[n_train:]
        net = backtest(prediction, returns)
        gross = (prediction.shift(1) * returns).sum(axis=1)
        turnover = prediction.shift(1).fillna(0.0).diff().abs().sum(axis=1)
        pnl_parts.append(net)
        gross_parts.append(gross)
        turnover_parts.append(turnover)
        fold_scores.append(float(net.mean() / net.std()))

    net = pd.concat(pnl_parts, ignore_index=True)
    gross = pd.concat(gross_parts, ignore_index=True)
    turnover = pd.concat(turnover_parts, ignore_index=True)
    scores = np.array(fold_scores)
    print(f"net_sharpe={net.mean() / net.std():+.6f}")
    print(f"gross_sharpe={gross.mean() / gross.std():+.6f}")
    print(f"mean_turnover={turnover.mean():.6f}")
    print(f"mean_cost={0.0005 * turnover.mean():.8f}")
    print(f"positive_folds={int((scores > 0).sum())}/{len(scores)}")
    print(f"fold_sharpe_std={scores.std():.6f}")
    print(f"train_seconds_mean={np.mean(train_times):.4f}")
    print(f"predict_seconds_mean={np.mean(predict_times):.4f}")

    latest = load_period(list_periods()[-1])
    n_val = min(800, int(0.2 * len(latest["features"])))
    n_train = len(latest["features"]) - n_val
    train_features = latest["features"].iloc[:n_train]
    train_target = latest["target"].iloc[:n_train]
    validation = latest["features"].iloc[n_train:]

    original = OrthogonalOrbitPredictor()
    original.train(train_features, train_target)
    first = original.predict(validation)
    second = original.predict(validation)
    print(f"demean_pass={check_cross_sectional_demean(first)}")
    print(f"repeat_max_abs_diff={np.max(np.abs(first.values - second.values)):.3e}")

    shuffled = np.random.default_rng(20260827).permutation(n_train)
    shuffled_model = OrthogonalOrbitPredictor()
    shuffled_model.train(train_features.iloc[shuffled], train_target.iloc[shuffled])
    shuffled_prediction = shuffled_model.predict(validation)
    print(f"shuffled_finite={np.isfinite(shuffled_prediction.values).all()}")
    print(f"shuffled_row_mean_max={np.abs(shuffled_prediction.mean(axis=1)).max():.3e}")

    zero_model = OrthogonalOrbitPredictor()
    zero_model.train(train_features, train_target * 0.0)
    zero_prediction = zero_model.predict(validation)
    print(f"zero_label_finite={np.isfinite(zero_prediction.values).all()}")
    print(f"zero_label_row_mean_max={np.abs(zero_prediction.mean(axis=1)).max():.3e}")


if __name__ == "__main__":
    main()
