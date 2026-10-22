"""Batch audit of causal formula families for PnL and published-city distance.

This is research code, not a submission.  Formulas use only same-row feature
cross-sections; the only time operation is an ordinary trailing EWM on the
signal used to control turnover.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from city_tools import compare_to_cities, compute_city, compute_gauge_fixed
from data_loader import list_periods, load_period
from evaluation import backtest


def demean(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=1, keepdims=True)


def rank_panel(frame: pd.DataFrame) -> np.ndarray:
    return (frame.rank(axis=1, method="average").to_numpy(dtype=float) - 10.5) / 9.5


def smooth(values: np.ndarray, span: int) -> np.ndarray:
    return demean(pd.DataFrame(values).ewm(span=span, adjust=False).mean().to_numpy())


def formulae(features: pd.DataFrame) -> dict[str, np.ndarray]:
    tickers = [f"ticker.{i}" for i in range(1, 21)]
    ranks = [rank_panel(features[f"Feature.{i}"][tickers]) for i in range(1, 7)]
    core = sum(ranks[:4]) / 4.0
    residuals = [item - core for item in ranks[:4]]
    disagreement = np.mean(np.square(np.stack(residuals, axis=2)), axis=2)
    f5, f6 = ranks[4], ranks[5]
    terms = {
        "core_contrarian": -core,
        "core_tail_short": -np.square(core),
        "core_tail_long": np.square(core),
        "core_cubic_short": -np.power(core, 3),
        "core_abs_short": -np.abs(core),
        "core_x_f5": core * f5,
        "core_x_f6": core * f6,
        "core_x_abs_f6": core * np.abs(f6),
        "f5_contrarian": -f5,
        "f5_tail_short": -np.square(f5),
        "f5_cubic_short": -np.power(f5, 3),
        "f5_abs_short": -np.abs(f5),
        "f6_trend": f6,
        "f6_tail_short": -np.square(f6),
        "f6_tail_long": np.square(f6),
        "f6_cubic_long": np.power(f6, 3),
        "f6_abs_short": -np.abs(f6),
        "f5_x_f6": f5 * f6,
        "disagreement_short": -disagreement,
        "residual_1_short": -residuals[0],
        "residual_2_short": -residuals[1],
        "residual_3_short": -residuals[2],
        "residual_4_short": -residuals[3],
        "difference_1_2": ranks[0] - ranks[1],
        "difference_3_4": ranks[2] - ranks[3],
    }
    for weight in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        terms[f"core_plus_{weight:g}_corexf6"] = -core + weight * core * f6
        terms[f"core_plus_{weight:g}_corexf5"] = -core + weight * core * f5
        terms[f"core_plus_{weight:g}_coretail"] = -core - weight * np.square(core)
        terms[f"core_plus_{weight:g}_f5cubic"] = -core - weight * np.power(f5, 3)
    return {name: demean(value) for name, value in terms.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--span", type=int, default=120)
    parser.add_argument("--periods", type=int, default=31)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--contains", type=str, default="")
    args = parser.parse_args()
    periods = list_periods()[-args.periods :]
    phrases = [part.strip() for part in args.contains.split(",") if part.strip()]
    pnl: dict[str, list[pd.Series]] = {}
    ic: dict[str, list[float]] = {}
    folds: dict[str, list[float]] = {}
    gauge_fixed: dict[str, list[np.ndarray]] = {}

    for period in periods:
        data = load_period(period)
        n_val = min(800, int(0.2 * len(data["features"])))
        tickers = [f"ticker.{i}" for i in range(1, 21)]
        returns = data["returns"].iloc[-n_val:][tickers]
        target = data["target"].iloc[-n_val:][tickers]
        gauge = np.load(f"data/train/gauge/{period}.small.gauge.npy")
        for name, raw in formulae(data["features"]).items():
            if phrases and not any(phrase in name for phrase in phrases):
                continue
            values = smooth(raw, args.span)[-n_val:]
            prediction = pd.DataFrame(values, index=returns.index, columns=tickers)
            series = backtest(prediction, returns)
            pnl.setdefault(name, []).append(series)
            folds.setdefault(name, []).append(float(series.mean() / series.std()))
            fixed = compute_gauge_fixed(values, gauge)
            gauge_fixed.setdefault(name, []).append(fixed)
            ic.setdefault(name, []).append(float(fixed[:, 0].mean()))
        print(f"{period}: completed")

    rows = []
    for name, pieces in pnl.items():
        joined = pd.concat(pieces, ignore_index=True)
        fixed = np.concatenate(gauge_fixed[name], axis=0)
        city = compute_city(fixed)
        comparison = compare_to_cities(city)
        rows.append({
            "name": name,
            "sharpe": float(joined.mean() / joined.std()),
            "ic": float(np.mean(ic[name])),
            "positive_folds": int(sum(value > 0 for value in folds[name])),
            "fold_std": float(np.std(folds[name])),
            "novelty": float(comparison.iloc[0]["angular_distance_deg"]),
            "nearest": str(comparison.iloc[0]["signal_id"]),
            "city": city,
        })
    report = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    pd.set_option("display.max_rows", None)
    print("=" * 108)
    print(report.drop(columns="city").head(args.top).to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print("=" * 108)
    print("Novel candidates (novelty >= 55 degrees):")
    print(
        report[report["novelty"] >= 55.0]
        .drop(columns="city")
        .sort_values("sharpe", ascending=False)
        .to_string(index=False, float_format=lambda x: f"{x:+.4f}")
    )


if __name__ == "__main__":
    main()
