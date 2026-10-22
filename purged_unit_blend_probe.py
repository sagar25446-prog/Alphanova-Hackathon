"""Screen normalized Prism/diversifier blends for novelty-aware candidates."""

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
    output = np.empty_like(values)
    for i in range(len(values)):
        state = alpha * values[i] + (1.0 - alpha) * state
        output[i] = state
    output -= output.mean(axis=1, keepdims=True)
    return output, state - state.mean()


def diversifier_from_ranks(rank_panel, n_train, name, span):
    (f1, f2, f3, f4, f5, f6) = rank_panel
    core = (f1 + f2 + f3 + f4) / 4.0
    raw = {"tail_core_f6": core**3 * f6, "core_f6": core * f6, "tail_core": -core**3}[name]
    raw -= raw.mean(axis=1, keepdims=True)
    _, state = smooth(raw[:n_train], span)
    val, _ = smooth(raw[n_train:], span, state)
    return val


def unit_rows(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def pooled(parts):
    x = pd.concat(parts, ignore_index=True)
    return float(x.mean() / x.std()) if x.std() else 0.0


def main():
    configs = [(name, span, weight) for name in ("tail_core_f6", "core_f6", "tail_core")
               for span in (120, 240, 480) for weight in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)]
    records = {key: {"pnl": [], "early": [], "late": [], "turnover": [], "gauge": []} for key in configs}
    for period in list_periods():
        data = load_period(period)
        n_val = min(800, int(0.2 * len(data["features"])))
        n_train = len(data["features"]) - n_val
        model = PurgedPrism()
        model.train(data["features"].iloc[:n_train], data["target"].iloc[:n_train])
        primary = unit_rows(model.predict(data["features"].iloc[n_train:]).to_numpy(float))
        tickers = sorted(data["returns"].columns, key=lambda name: int(name.split(".")[-1]))
        ret = data["returns"].iloc[n_train:][tickers]
        gauge = np.load(f"data/train/gauge/{period}.small.gauge.npy")
        rank_panel, _ = ranks(data["features"])
        legs = {(name, span): diversifier_from_ranks(rank_panel, n_train, name, span) for name, span, _ in configs}
        for name, span, weight in configs:
            right = unit_rows(legs[(name, span)])
            values = primary + weight * right
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
    for key, v in records.items():
        city = compute_city(np.concatenate(v["gauge"], axis=0))
        near = compare_to_cities(city, CURRENT_MAP).iloc[0]
        rows.append({
            "name": key[0], "span": key[1], "weight": key[2],
            "early_net": pooled(v["early"]), "late_net": pooled(v["late"]),
            "full_net": pooled(v["pnl"]), "late_pos": sum(float(x.mean()/x.std()) > 0 for x in v["late"]),
            "turnover": float(np.mean(v["turnover"])), "city_deg": float(near["angular_distance_deg"]),
            "nearest": str(near["signal_id"]),
        })
    out = pd.DataFrame(rows)
    out.to_csv("purged_unit_blend_screen.csv", index=False)
    print("TOP_FULL")
    print(out.sort_values("full_net", ascending=False).head(25).to_string(index=False))
    print("NOVEL_AND_POSITIVE")
    print(out[(out.city_deg >= 60) & (out.late_net > 0)].sort_values("full_net", ascending=False).head(30).to_string(index=False))


if __name__ == "__main__":
    main()
