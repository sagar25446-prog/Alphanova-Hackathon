"""Offline-only factor screen with a current-city navigation report.

This never reads the city file while generating model predictions.  Formulas and
both orientations are fixed before scoring; for each formula the orientation is
chosen on walk-forward periods 001--019, while 020--031 is a one-time holdout.
"""

import csv

import numpy as np
import pandas as pd

from city_tools import compare_to_cities, compute_city, compute_gauge_fixed
from data_loader import list_periods, load_period
from evaluation import backtest


CURRENT_MAP = "current_city_map_20260827/signal_cities_ambient_coords.parquet"
SPAN = 120
FORMULAS = (
    "core", "core_cube", "core_square", "disagreement", "f5", "f6",
    "f1_f2", "f1_f3", "f1_f4", "f2_f3", "f2_f4", "f3_f4",
    "f5_f6", "f5sq_f6", "f5_f6sq", "f5cube_minus_f6cube",
    "tailcore_f5", "tailcore_f6", "core2_f5", "core2_f6",
    "core_f5_f6", "asym12_f5", "asym34_f6",
    "spread12_spread34", "core_f5", "core_f6", "prism_base",
)


def pooled_sharpe(parts: list[pd.Series]) -> float:
    joined = pd.concat(parts, ignore_index=True)
    return float(joined.mean() / joined.std()) if joined.std() else 0.0


def smooth(values: np.ndarray, initial: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    alpha = 2.0 / (SPAN + 1.0)
    previous = np.zeros(values.shape[1], dtype=float) if initial is None else initial.copy()
    output = np.empty_like(values)
    for row in range(len(values)):
        previous = alpha * values[row] + (1.0 - alpha) * previous
        output[row] = previous
    return output - output.mean(axis=1, keepdims=True), previous - previous.mean()


def get_ranks(features: pd.DataFrame) -> tuple[list[np.ndarray], list[str]]:
    tickers = sorted(features["Feature.1"].columns, key=lambda name: int(name.split(".")[-1]))
    ranks = []
    for number in range(1, 7):
        values = np.nan_to_num(features[f"Feature.{number}"][tickers].to_numpy(float))
        order = np.argsort(np.argsort(values, axis=1, kind="mergesort"), axis=1, kind="mergesort")
        ranks.append((order.astype(float) - 9.5) / 9.5)
    return ranks, tickers


def formula_panel(features: pd.DataFrame, name: str) -> tuple[np.ndarray, list[str]]:
    (f1, f2, f3, f4, f5, f6), tickers = get_ranks(features)
    core = (f1 + f2 + f3 + f4) / 4.0
    core2 = core * core - np.mean(core * core, axis=1, keepdims=True)
    disagreement = np.mean(np.square(np.stack([f1 - core, f2 - core, f3 - core, f4 - core], axis=2)), axis=2)
    pieces = {
        "core": core,
        "core_cube": core ** 3,
        "core_square": core2,
        "disagreement": disagreement,
        "f5": f5,
        "f6": f6,
        "f1_f2": f1 * f2,
        "f1_f3": f1 * f3,
        "f1_f4": f1 * f4,
        "f2_f3": f2 * f3,
        "f2_f4": f2 * f4,
        "f3_f4": f3 * f4,
        "f5_f6": f5 * f6,
        "f5sq_f6": (f5 * f5 - np.mean(f5 * f5, axis=1, keepdims=True)) * f6,
        "f5_f6sq": f5 * (f6 * f6 - np.mean(f6 * f6, axis=1, keepdims=True)),
        "f5cube_minus_f6cube": f5 ** 3 - f6 ** 3,
        "tailcore_f5": core ** 3 * f5,
        "tailcore_f6": core ** 3 * f6,
        "core2_f5": core2 * f5,
        "core2_f6": core2 * f6,
        "core_f5_f6": core * f5 * f6,
        "asym12_f5": (f1 * f1 - f2 * f2) * f5,
        "asym34_f6": (f3 * f3 - f4 * f4) * f6,
        "spread12_spread34": (f1 - f2) * (f3 - f4),
        "core_f5": core * f5,
        "core_f6": core * f6,
        "prism_base": -core ** 3 + core * f6,
    }
    values = pieces[name]
    return values - values.mean(axis=1, keepdims=True), tickers


def main() -> None:
    outcomes: dict[tuple[str, int], dict[str, list]] = {
        (name, sign): {"pnl": [], "early": [], "late": [], "gauge": [], "turnover": []}
        for name in FORMULAS for sign in (-1, 1)
    }
    for period in list_periods():
        data = load_period(period)
        n_val = min(800, int(0.2 * len(data["features"])))
        n_train = len(data["features"]) - n_val
        ret = data["returns"].iloc[n_train:]
        gauge = np.load(f"data/train/gauge/{period}.small.gauge.npy")
        for name in FORMULAS:
            raw, tickers = formula_panel(data["features"], name)
            _, initial = smooth(raw[:n_train])
            val, _ = smooth(raw[n_train:], initial)
            for sign in (-1, 1):
                values = sign * val
                pred = pd.DataFrame(values, index=ret.index, columns=tickers)
                pnl = backtest(pred, ret[tickers])
                bucket = outcomes[(name, sign)]
                bucket["pnl"].append(pnl)
                bucket["early" if int(period) <= 19 else "late"].append(pnl)
                bucket["gauge"].append(compute_gauge_fixed(values, gauge))
                bucket["turnover"].append(pred.shift(1).fillna(0.0).diff().abs().sum(axis=1))
        print(f"screened {period}", flush=True)

    rows = []
    for name in FORMULAS:
        sign = max((-1, 1), key=lambda item: pooled_sharpe(outcomes[(name, item)]["early"]))
        metrics = outcomes[(name, sign)]
        city = compute_city(np.concatenate(metrics["gauge"], axis=0))
        nearest = compare_to_cities(city, CURRENT_MAP).iloc[0]
        rows.append({
            "formula": name,
            "selected_sign_001_019": sign,
            "early_net": pooled_sharpe(metrics["early"]),
            "late_net": pooled_sharpe(metrics["late"]),
            "full_net": pooled_sharpe(metrics["pnl"]),
            "turnover": float(pd.concat(metrics["turnover"], ignore_index=True).mean()),
            "city_deg_current": float(nearest["angular_distance_deg"]),
            "nearest_current_signal": str(nearest["signal_id"]),
        })
    rows.sort(key=lambda row: row["city_deg_current"], reverse=True)
    with open("current_city_formula_screen.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print("TOP_BY_CURRENT_CITY")
    print(pd.DataFrame(rows).head(27).to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    robust = [row for row in rows if row["early_net"] > 0 and row["late_net"] > 0]
    robust.sort(key=lambda row: row["city_deg_current"], reverse=True)
    print("ROBUST_POSITIVE_BOTH_BLOCKS")
    print(pd.DataFrame(robust).to_string(index=False, float_format=lambda x: f"{x:+.4f}"))


if __name__ == "__main__":
    main()
