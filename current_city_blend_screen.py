"""Offline navigation screen against the organiser's current published city map.

The city coordinates are used only after predictions are generated, to rank
research candidates for a later official novelty check.  They are never inputs
to any model or formula.
"""

import csv

import numpy as np
import pandas as pd

from alphanova_causal_prism import CausalPrismPredictor
from city_tools import compare_to_cities, compute_city, compute_gauge_fixed
from data_loader import list_periods, load_period
from evaluation import backtest


CURRENT_MAP = "current_city_map_20260827/signal_cities_ambient_coords.parquet"


def ratio(parts: list[pd.Series]) -> float:
    values = pd.concat(parts, ignore_index=True)
    return float(values.mean() / values.std()) if values.std() else 0.0


def unit_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def smooth(values: np.ndarray, span: int, initial: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    alpha = 2.0 / (span + 1.0)
    state = np.zeros(values.shape[1], dtype=float) if initial is None else initial.copy()
    output = np.empty_like(values)
    for row in range(len(values)):
        state = alpha * values[row] + (1.0 - alpha) * state
        output[row] = state
    return output - output.mean(axis=1, keepdims=True), state - state.mean()


def macd(values: np.ndarray, fast: int, slow: int, initial: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    fast_initial = None if initial is None else initial[0]
    slow_initial = None if initial is None else initial[1]
    quick, fast_state = smooth(values, fast, fast_initial)
    slow_values, slow_state = smooth(values, slow, slow_initial)
    return quick - slow_values, (fast_state, slow_state)


def ranks(features: pd.DataFrame) -> tuple[list[np.ndarray], list[str]]:
    tickers = sorted(features["Feature.1"].columns, key=lambda name: int(name.split(".")[-1]))
    output = []
    for number in range(1, 7):
        values = np.nan_to_num(features[f"Feature.{number}"][tickers].to_numpy(float))
        order = np.argsort(np.argsort(values, axis=1, kind="mergesort"), axis=1, kind="mergesort")
        output.append((order.astype(float) - 9.5) / 9.5)
    return output, tickers


def diversifiers(features: pd.DataFrame, n_train: int) -> tuple[dict[str, np.ndarray], list[str]]:
    rank, tickers = ranks(features)
    core = sum(rank[:4]) / 4.0
    raw = {
        "tail_core": -core ** 3,
        "tail_core_f6": core ** 3 * rank[5],
        "core_level": -core,
        "f5_level": -rank[4],
    }
    output = {}
    for name, values in raw.items():
        values = values - values.mean(axis=1, keepdims=True)
        _, state = smooth(values[:n_train], 120)
        val, _ = smooth(values[n_train:], 120, state)
        output[name] = val
    level = raw["core_level"]
    _, state = macd(level[:n_train], 48, 168)
    val, _ = macd(level[n_train:], 48, 168, state)
    output["core_macd_48_168"] = val - val.mean(axis=1, keepdims=True)
    level = raw["f5_level"]
    _, state = macd(level[:n_train], 48, 168)
    val, _ = macd(level[n_train:], 48, 168, state)
    output["f5_macd_48_168"] = val - val.mean(axis=1, keepdims=True)
    return output, tickers


def main() -> None:
    configs = [("raw", weight) for weight in (0.25, 0.5, 1.0, 2.0, 4.0)]
    configs += [("unit", weight) for weight in (0.10, 0.25, 0.5, 1.0, 2.0)]
    records: dict[tuple[str, str, float], dict[str, list]] = {}
    for period in list_periods():
        data = load_period(period)
        n_val = min(800, int(0.2 * len(data["features"])))
        n_train = len(data["features"]) - n_val
        prism = CausalPrismPredictor()
        prism.train(data["features"].iloc[:n_train], data["target"].iloc[:n_train])
        primary = prism.predict(data["features"].iloc[n_train:])
        diversifier_map, tickers = diversifiers(data["features"], n_train)
        ret = data["returns"].iloc[n_train:][tickers]
        gauge = np.load(f"data/train/gauge/{period}.small.gauge.npy")
        for name, leg in diversifier_map.items():
            for scale_mode, weight in configs:
                left = primary[tickers].to_numpy(float)
                right = leg
                if scale_mode == "unit":
                    left = unit_rows(left)
                    right = unit_rows(right)
                values = left + weight * right
                values -= values.mean(axis=1, keepdims=True)
                pred = pd.DataFrame(values, index=ret.index, columns=tickers)
                pnl = backtest(pred, ret)
                key = (name, scale_mode, weight)
                row = records.setdefault(key, {"pnl": [], "early": [], "late": [], "gauge": [], "turnover": []})
                row["pnl"].append(pnl)
                row["turnover"].append(pred.shift(1).fillna(0.0).diff().abs().sum(axis=1))
                row["gauge"].append(compute_gauge_fixed(values, gauge))
                row["early" if int(period) <= 19 else "late"].append(pnl)
        print(f"screened {period}", flush=True)
    rows = []
    for (name, scale_mode, weight), values in records.items():
        city = compute_city(np.concatenate(values["gauge"], axis=0))
        nearest = compare_to_cities(city, CURRENT_MAP).iloc[0]
        rows.append({
            "diversifier": name,
            "scale": scale_mode,
            "weight": weight,
            "city_deg_current": float(nearest["angular_distance_deg"]),
            "nearest_current_signal": str(nearest["signal_id"]),
            "early_net": ratio(values["early"]),
            "late_net": ratio(values["late"]),
            "full_net": ratio(values["pnl"]),
            "turnover": float(pd.concat(values["turnover"], ignore_index=True).mean()),
        })
    rows.sort(key=lambda row: (row["city_deg_current"] >= 60.0, row["early_net"], row["late_net"]), reverse=True)
    with open("current_city_blend_screen.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(pd.DataFrame(rows).head(25).to_string(index=False, float_format=lambda x: f"{x:+.4f}"))


if __name__ == "__main__":
    main()
