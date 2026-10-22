"""Focused novelty/Sharpe path for the purge-aware Prism plus tail-core-F6."""

from __future__ import annotations

import numpy as np
import pandas as pd

from blend_purged_probe import PurgedPrism
from city_tools import compare_to_cities, compute_city, compute_gauge_fixed
from data_loader import list_periods, load_period
from evaluation import backtest

CURRENT_MAP = "current_city_map_20260827/signal_cities_ambient_coords.parquet"


def ranks(features):
    tickers = sorted(features["Feature.1"].columns, key=lambda name: int(name.split(".")[-1]))
    out = []
    for number in range(1, 7):
        v = np.nan_to_num(features[f"Feature.{number}"][tickers].to_numpy(float))
        order = np.argsort(np.argsort(v, axis=1, kind="mergesort"), axis=1, kind="mergesort")
        out.append((order.astype(float) - 9.5) / 9.5)
    return out, tickers


def smooth(values, span, initial=None):
    alpha = 2.0 / (span + 1.0)
    state = np.zeros(values.shape[1]) if initial is None else initial.copy()
    out = np.empty_like(values)
    for i in range(len(values)):
        state = alpha * values[i] + (1.0 - alpha) * state
        out[i] = state
    out -= out.mean(axis=1, keepdims=True)
    return out, state - state.mean()


def pooled(parts):
    x = pd.concat(parts, ignore_index=True)
    return float(x.mean() / x.std()) if x.std() else 0.0


def main():
    # Predeclared path around the previously identified transition region.
    configs = [("raw", w) for w in (4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)]
    configs += [("unit", w) for w in (0.75, 1.0, 1.25, 1.5, 1.75, 2.0)]
    rec = {key: {"pnl": [], "early": [], "late": [], "turnover": [], "gauge": []} for key in configs}
    for period in list_periods():
        data = load_period(period)
        n_val = min(800, int(0.2 * len(data["features"])))
        n_train = len(data["features"]) - n_val
        model = PurgedPrism()
        model.train(data["features"].iloc[:n_train], data["target"].iloc[:n_train])
        primary = model.predict(data["features"].iloc[n_train:]).to_numpy(float).copy()
        if not np.isfinite(primary).all():
            raise ValueError("non-finite primary")
        primary_unit = primary / np.maximum(np.linalg.norm(primary, axis=1, keepdims=True), 1e-12)
        (f1, f2, f3, f4, f5, f6), tickers = ranks(data["features"])
        core = (f1 + f2 + f3 + f4) / 4.0
        raw = core**3 * f6
        raw -= raw.mean(axis=1, keepdims=True)
        _, state = smooth(raw[:n_train], 480)
        leg, _ = smooth(raw[n_train:], 480, state)
        leg_unit = leg / np.maximum(np.linalg.norm(leg, axis=1, keepdims=True), 1e-12)
        ret = data["returns"].iloc[n_train:][tickers]
        gauge = np.load(f"data/train/gauge/{period}.small.gauge.npy")
        for mode, weight in configs:
            left = primary if mode == "raw" else primary_unit
            right = leg if mode == "raw" else leg_unit
            values = left + weight * right
            values -= values.mean(axis=1, keepdims=True)
            pred = pd.DataFrame(values, index=ret.index, columns=tickers)
            pnl = backtest(pred, ret)
            bucket = rec[(mode, weight)]
            bucket["pnl"].append(pnl)
            bucket["early" if int(period) <= 19 else "late"].append(pnl)
            bucket["turnover"].append(float(pred.shift(1).fillna(0.0).diff().abs().sum(axis=1).mean()))
            bucket["gauge"].append(compute_gauge_fixed(values, gauge))
        print(f"screened {period}", flush=True)
    rows = []
    for key, v in rec.items():
        city = compute_city(np.concatenate(v["gauge"], axis=0))
        near = compare_to_cities(city, CURRENT_MAP).iloc[0]
        rows.append({
            "mode": key[0], "weight": key[1], "early_net": pooled(v["early"]),
            "late_net": pooled(v["late"]), "full_net": pooled(v["pnl"]),
            "late_pos": sum(float(x.mean()/x.std()) > 0 for x in v["late"]),
            "turnover": float(np.mean(v["turnover"])), "city_deg": float(near["angular_distance_deg"]),
            "nearest": str(near["signal_id"]),
        })
    out = pd.DataFrame(rows)
    out.to_csv("purged_tail_path_screen.csv", index=False)
    print(out.sort_values(["city_deg", "full_net"], ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
