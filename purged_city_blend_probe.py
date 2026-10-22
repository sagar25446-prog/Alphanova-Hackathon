"""Causal, purge-aware Prism/diversifier screen with current city distances."""

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
    output = []
    for number in range(1, 7):
        values = np.nan_to_num(features[f"Feature.{number}"][tickers].to_numpy(float))
        order = np.argsort(np.argsort(values, axis=1, kind="mergesort"), axis=1, kind="mergesort")
        output.append((order.astype(float) - 9.5) / 9.5)
    return output, tickers


def smooth(values, span, initial=None):
    alpha = 2.0 / (span + 1.0)
    state = np.zeros(values.shape[1], dtype=float) if initial is None else initial.copy()
    out = np.empty_like(values)
    for row in range(len(values)):
        state = alpha * values[row] + (1.0 - alpha) * state
        out[row] = state
    out -= out.mean(axis=1, keepdims=True)
    return out, state - state.mean()


def diversifier_values(features, n_train, name, span):
    (f1, f2, f3, f4, f5, f6), tickers = ranks(features)
    core = (f1 + f2 + f3 + f4) / 4.0
    raw = {
        "tail_core": -core**3,
        "tail_core_f6": core**3 * f6,
        "core_level": -core,
        "f5_level": -f5,
    }[name]
    raw -= raw.mean(axis=1, keepdims=True)
    _, state = smooth(raw[:n_train], span)
    val, _ = smooth(raw[n_train:], span, state)
    return val, tickers


def pooled(parts):
    x = pd.concat(parts, ignore_index=True)
    return float(x.mean() / x.std()) if x.std() else 0.0


def main():
    configs = [("none", 0, 0.0)]
    configs += [(name, span, weight) for name in ("tail_core", "tail_core_f6", "core_level", "f5_level")
                for span in (120, 240, 480) for weight in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)]
    records = {key: {"pnl": [], "early": [], "late": [], "turnover": [], "gauge": []} for key in configs}
    for period in list_periods():
        data = load_period(period)
        n_val = min(800, int(0.2 * len(data["features"])))
        n_train = len(data["features"]) - n_val
        primary_model = PurgedPrism()
        primary_model.train(data["features"].iloc[:n_train], data["target"].iloc[:n_train])
        primary = primary_model.predict(data["features"].iloc[n_train:])
        tickers = primary.columns.tolist()
        ret = data["returns"].iloc[n_train:][tickers]
        gauge = np.load(f"data/train/gauge/{period}.small.gauge.npy")
        diversifiers = {}
        for name in ("tail_core", "tail_core_f6", "core_level", "f5_level"):
            for span in (120, 240, 480):
                diversifiers[(name, span)], _ = diversifier_values(data["features"], n_train, name, span)
        for name, span, weight in configs:
            left = primary.to_numpy(float).copy()
            if name == "none":
                values = left
            else:
                values = left + weight * diversifiers[(name, span)]
            values -= values.mean(axis=1, keepdims=True)
            pred = pd.DataFrame(values, index=ret.index, columns=tickers)
            pnl = backtest(pred, ret)
            bucket = records[(name, span, weight)]
            bucket["pnl"].append(pnl)
            bucket["early" if int(period) <= 19 else "late"].append(pnl)
            bucket["turnover"].append(float(pred.shift(1).fillna(0.0).diff().abs().sum(axis=1).mean()))
            bucket["gauge"].append(compute_gauge_fixed(values, gauge))
        print(f"screened {period}", flush=True)
    rows = []
    for key, values in records.items():
        city = compute_city(np.concatenate(values["gauge"], axis=0))
        nearest = compare_to_cities(city, CURRENT_MAP).iloc[0]
        late = values["late"]
        rows.append({
            "name": key[0], "span": key[1], "weight": key[2],
            "early_net": pooled(values["early"]), "late_net": pooled(late),
            "full_net": pooled(values["pnl"]),
            "late_pos": sum(float(x.mean() / x.std()) > 0 for x in late),
            "turnover": float(np.mean(values["turnover"])),
            "city_deg": float(nearest["angular_distance_deg"]),
            "nearest": str(nearest["signal_id"]),
        })
    out = pd.DataFrame(rows)
    out.to_csv("purged_city_blend_screen.csv", index=False)
    print("TOP_FULL")
    print(out.sort_values("full_net", ascending=False).head(20).to_string(index=False))
    print("NOVEL_AND_POSITIVE")
    print(out[(out.city_deg >= 60) & (out.late_net > 0)].sort_values("full_net", ascending=False).head(30).to_string(index=False))


if __name__ == "__main__":
    main()
