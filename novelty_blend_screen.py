"""Offline only: predeclared performance/geometry screen for a two-leg blend."""

from __future__ import annotations

import csv

import numpy as np
import pandas as pd

from alphanova_causal_prism import CausalPrismPredictor
from alphanova_core_cross_orbit import CoreCrossOrbitPredictor
from city_tools import compare_to_cities, compute_city, compute_gauge_fixed
from data_loader import list_periods, load_period


CONFIGS = (
    ("direct_0", 0.0, False),
    ("direct_025", 0.25, False),
    ("direct_050", 0.50, False),
    ("direct_100", 1.00, False),
    ("direct_200", 2.00, False),
    ("direct_400", 4.00, False),
    ("norm_025", 0.25, True),
    ("norm_050", 0.50, True),
    ("norm_100", 1.00, True),
    ("norm_200", 2.00, True),
    ("norm_400", 4.00, True),
)


def net_returns(predictions: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    pnl = (predictions.shift(1) * returns).sum(axis=1)
    turnover = predictions.shift(1).fillna(0.0).diff().abs().sum(axis=1)
    pnl = pnl - 0.0005 * turnover
    pnl.iloc[0] = 0.0
    return pnl


def row_unit(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def screen() -> None:
    results = {
        name: {"pnl": [], "early": [], "late": [], "turnover": [], "gauge": []}
        for name, _, _ in CONFIGS
    }
    for period in list_periods():
        data = load_period(period)
        n_val = min(800, int(0.2 * len(data["features"])))
        n_train = len(data["features"]) - n_val
        prism = CausalPrismPredictor()
        cross = CoreCrossOrbitPredictor()
        prism.train(data["features"].iloc[:n_train], data["target"].iloc[:n_train])
        cross.train(data["features"].iloc[:n_train], data["target"].iloc[:n_train])
        first = prism.predict(data["features"].iloc[n_train:])
        second = cross.predict(data["features"].iloc[n_train:])[first.columns]
        gauge = np.load(f"data/train/gauge/{period}.small.gauge.npy")
        returns = data["returns"].iloc[n_train:][first.columns]
        for name, weight, normalize in CONFIGS:
            left = row_unit(first.values) if normalize else first.values
            right = row_unit(second.values) if normalize else second.values
            values = left + weight * right
            values -= values.mean(axis=1, keepdims=True)
            prediction = pd.DataFrame(values, index=first.index, columns=first.columns)
            pnl = net_returns(prediction, returns)
            results[name]["pnl"].append(pnl)
            results[name]["turnover"].append(float(prediction.shift(1).fillna(0.0).diff().abs().sum(axis=1).mean()))
            results[name]["gauge"].append(compute_gauge_fixed(values, gauge))
            results[name]["early" if int(period) <= 19 else "late"].append(float(pnl.mean() / pnl.std()))
        print(f"screened period {period}", flush=True)
    rows = []
    for name, _, _ in CONFIGS:
        values = results[name]
        pnl = pd.concat(values["pnl"], ignore_index=True)
        city = compute_city(np.concatenate(values["gauge"], axis=0))
        nearest = compare_to_cities(city).iloc[0]
        rows.append({
            "name": name,
            "net_sharpe": float(pnl.mean() / pnl.std()),
            "early_mean_fold_sharpe": float(np.mean(values["early"])),
            "late_mean_fold_sharpe": float(np.mean(values["late"])),
            "late_positive": int(np.sum(np.array(values["late"]) > 0)),
            "mean_turnover": float(np.mean(values["turnover"])),
            "city_novelty_deg": float(nearest["angular_distance_deg"]),
            "nearest_city": str(nearest["signal_id"]),
        })
    rows.sort(key=lambda row: row["city_novelty_deg"], reverse=True)
    with open("novelty_blend_screen.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:+.5f}"))


if __name__ == "__main__":
    screen()
