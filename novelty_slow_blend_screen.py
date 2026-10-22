"""Offline only: fixed slow-diversifier blend screen."""

from __future__ import annotations

import csv

import numpy as np
import pandas as pd

from alphanova_causal_prism import CausalPrismPredictor
from alphanova_core_cross_orbit import CoreCrossOrbitPredictor
from city_tools import compare_to_cities, compute_city, compute_gauge_fixed
from data_loader import list_periods, load_period


CONFIGS = tuple((span, weight) for span in (120, 240, 480) for weight in (2.0, 4.0, 8.0))


def net_returns(predictions: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    pnl = (predictions.shift(1) * returns).sum(axis=1)
    turnover = predictions.shift(1).fillna(0.0).diff().abs().sum(axis=1)
    pnl = pnl - 0.0005 * turnover
    pnl.iloc[0] = 0.0
    return pnl


def screen() -> None:
    results = {
        (span, weight): {"pnl": [], "early": [], "late": [], "turnover": [], "gauge": []}
        for span, weight in CONFIGS
    }
    for period in list_periods():
        data = load_period(period)
        n_val = min(800, int(0.2 * len(data["features"])))
        n_train = len(data["features"]) - n_val
        prism = CausalPrismPredictor()
        prism.train(data["features"].iloc[:n_train], data["target"].iloc[:n_train])
        first = prism.predict(data["features"].iloc[n_train:])
        gauge = np.load(f"data/train/gauge/{period}.small.gauge.npy")
        returns = data["returns"].iloc[n_train:][first.columns]
        for span, weight in CONFIGS:
            cross = CoreCrossOrbitPredictor()
            cross.span = span
            cross.train(data["features"].iloc[:n_train], data["target"].iloc[:n_train])
            second = cross.predict(data["features"].iloc[n_train:])[first.columns]
            raw = first.values + weight * second.values
            raw -= raw.mean(axis=1, keepdims=True)
            prediction = pd.DataFrame(raw, index=first.index, columns=first.columns)
            pnl = net_returns(prediction, returns)
            result = results[(span, weight)]
            result["pnl"].append(pnl)
            result["turnover"].append(float(prediction.shift(1).fillna(0.0).diff().abs().sum(axis=1).mean()))
            result["gauge"].append(compute_gauge_fixed(raw, gauge))
            result["early" if int(period) <= 19 else "late"].append(float(pnl.mean() / pnl.std()))
        print(f"screened period {period}", flush=True)
    rows = []
    for span, weight in CONFIGS:
        result = results[(span, weight)]
        pnl = pd.concat(result["pnl"], ignore_index=True)
        city = compute_city(np.concatenate(result["gauge"], axis=0))
        nearest = compare_to_cities(city).iloc[0]
        rows.append({
            "diversifier_span": span,
            "blend_weight": weight,
            "net_sharpe": float(pnl.mean() / pnl.std()),
            "early_mean_fold_sharpe": float(np.mean(result["early"])),
            "late_mean_fold_sharpe": float(np.mean(result["late"])),
            "late_positive": int(np.sum(np.array(result["late"]) > 0)),
            "mean_turnover": float(np.mean(result["turnover"])),
            "city_novelty_deg": float(nearest["angular_distance_deg"]),
            "nearest_city": str(nearest["signal_id"]),
        })
    rows.sort(key=lambda row: row["city_novelty_deg"], reverse=True)
    with open("novelty_slow_blend_screen.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:+.5f}"))


if __name__ == "__main__":
    screen()
