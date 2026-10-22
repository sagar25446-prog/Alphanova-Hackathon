"""Metrics report for the local AlphaNova candidate; not a submission."""

import time

import numpy as np
import pandas as pd

from alphanova_causal_prism import CausalPrismPredictor
from data_loader import list_periods, load_period
from evaluation import backtest


def main() -> None:
    pnl_parts = []
    gross_parts = []
    turnover_parts = []
    ic_parts = []
    fold_rows = []
    train_times = []
    predict_times = []
    for period in list_periods():
        data = load_period(period)
        n_val = min(800, int(0.2 * len(data["features"])))
        n_train = len(data["features"]) - n_val
        predictor = CausalPrismPredictor()
        started = time.perf_counter()
        predictor.train(data["features"].iloc[:n_train], data["target"].iloc[:n_train])
        train_times.append(time.perf_counter() - started)
        started = time.perf_counter()
        prediction = predictor.predict(data["features"].iloc[n_train:])
        predict_times.append(time.perf_counter() - started)
        returns = data["returns"].iloc[n_train:]
        target = data["target"].iloc[n_train:]
        net = backtest(prediction, returns)
        gross = (prediction.shift(1) * returns).sum(axis=1)
        turnover = prediction.shift(1).fillna(0.0).diff().abs().sum(axis=1)
        fold_sharpe = float(net.mean() / net.std())
        fold_ic = float(prediction.corrwith(target, axis=1).mean())
        fold_rows.append((period, fold_sharpe, fold_ic, float(turnover.mean())))
        pnl_parts.append(net)
        gross_parts.append(gross)
        turnover_parts.append(turnover)
        ic_parts.append(fold_ic)
    net = pd.concat(pnl_parts, ignore_index=True)
    gross = pd.concat(gross_parts, ignore_index=True)
    turnover = pd.concat(turnover_parts, ignore_index=True)
    scores = np.array([row[1] for row in fold_rows])
    print(f"net_sharpe={net.mean() / net.std():+.6f}")
    print(f"gross_sharpe={gross.mean() / gross.std():+.6f}")
    print(f"mean_ic={np.mean(ic_parts):+.6f}")
    print(f"mean_turnover={turnover.mean():.6f}")
    print(f"mean_cost={0.0005 * turnover.mean():.8f}")
    print(f"positive_folds={int((scores > 0).sum())}/{len(scores)}")
    print(f"fold_sharpe_std={scores.std():.6f}")
    print(f"train_seconds_mean={np.mean(train_times):.4f}")
    print(f"predict_seconds_mean={np.mean(predict_times):.4f}")
    print("folds")
    for period, sharpe, ic, value in fold_rows:
        print(f"{period},{sharpe:+.5f},{ic:+.5f},{value:.6f}")


if __name__ == "__main__":
    main()
