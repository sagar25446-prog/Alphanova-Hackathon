"""Offline research only: fixed nonlinear factor screen with a held-out late window.

This is not a submission.  It contains no predictor and is used only to decide
whether an economically distinct feature family deserves a fully compliant
submission implementation.
"""

from __future__ import annotations

import csv

import numpy as np
import pandas as pd

from data_loader import list_periods, load_period


CONFIGS = (
    "f5_f6",
    "f5sq_f6",
    "f5_f6sq",
    "f5cube_minus_f6cube",
    "f5_tail_f6_tail",
    "f5_absf6",
    "f6_absf5",
    "f5sq_minus_f6sq",
    "f1_f2",
    "f1_f3",
    "f1_f4",
    "f2_f3",
    "f2_f4",
    "f3_f4",
    "f1f2_minus_f3f4",
    "spread12_spread34",
    "tailcore_f5",
    "tailcore_f6",
    "core2_f5",
    "core2_f6",
    "core_f5_f6",
    "f1f2f3",
    "f2f3f4",
    "f1f3_f5",
    "f2f4_f6",
    "asym12_f5",
    "asym34_f6",
)


def ranks(features: pd.DataFrame) -> np.ndarray:
    tickers = sorted(features["Feature.1"].columns, key=lambda x: int(x.split(".")[-1]))
    output = []
    for number in range(1, 7):
        values = features[f"Feature.{number}"][tickers].to_numpy(dtype=float)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        ordered = pd.DataFrame(values).rank(axis=1, method="average").to_numpy(dtype=float)
        output.append((ordered - 10.5) / 9.5)
    return np.stack(output, axis=0)


def factor(r: np.ndarray, name: str) -> np.ndarray:
    f1, f2, f3, f4, f5, f6 = r
    core = (f1 + f2 + f3 + f4) / 4.0
    tail = core ** 3
    centered_square_5 = f5 ** 2 - np.mean(f5 ** 2, axis=1, keepdims=True)
    centered_square_6 = f6 ** 2 - np.mean(f6 ** 2, axis=1, keepdims=True)
    formulas = {
        "f5_f6": f5 * f6,
        "f5sq_f6": centered_square_5 * f6,
        "f5_f6sq": f5 * centered_square_6,
        "f5cube_minus_f6cube": f5 ** 3 - f6 ** 3,
        "f5_tail_f6_tail": f5 ** 3 * f6 - f5 * f6 ** 3,
        "f5_absf6": f5 * (np.abs(f6) - np.mean(np.abs(f6), axis=1, keepdims=True)),
        "f6_absf5": f6 * (np.abs(f5) - np.mean(np.abs(f5), axis=1, keepdims=True)),
        "f5sq_minus_f6sq": centered_square_5 - centered_square_6,
        "f1_f2": f1 * f2,
        "f1_f3": f1 * f3,
        "f1_f4": f1 * f4,
        "f2_f3": f2 * f3,
        "f2_f4": f2 * f4,
        "f3_f4": f3 * f4,
        "f1f2_minus_f3f4": f1 * f2 - f3 * f4,
        "spread12_spread34": (f1 - f2) * (f3 - f4),
        "tailcore_f5": tail * f5,
        "tailcore_f6": tail * f6,
        "core2_f5": (core ** 2 - np.mean(core ** 2, axis=1, keepdims=True)) * f5,
        "core2_f6": (core ** 2 - np.mean(core ** 2, axis=1, keepdims=True)) * f6,
        "core_f5_f6": core * f5 * f6,
        "f1f2f3": f1 * f2 * f3,
        "f2f3f4": f2 * f3 * f4,
        "f1f3_f5": f1 * f3 * f5,
        "f2f4_f6": f2 * f4 * f6,
        "asym12_f5": (f1 ** 2 - f2 ** 2) * f5,
        "asym34_f6": (f3 ** 2 - f4 ** 2) * f6,
    }
    values = formulas[name]
    return values - np.mean(values, axis=1, keepdims=True)


def smooth(values: np.ndarray, span: int, initial: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    alpha = 2.0 / (span + 1.0)
    previous = np.zeros(values.shape[1], dtype=float) if initial is None else initial.copy()
    output = np.empty_like(values)
    for row in range(len(values)):
        previous = alpha * values[row] + (1.0 - alpha) * previous
        output[row] = previous
    output -= output.mean(axis=1, keepdims=True)
    return output, previous - previous.mean()


def net_returns(predictions: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """Exact competition portfolio accounting, kept local to this research file."""
    pnl = (predictions.shift(1) * returns).sum(axis=1)
    turnover = predictions.shift(1).fillna(0.0).diff().abs().sum(axis=1)
    pnl = pnl - 0.0005 * turnover
    pnl.iloc[0] = 0.0
    return pnl


def screen() -> None:
    results: dict[str, dict[str, list[float]]] = {
        name: {"early": [], "late": [], "all": [], "turnover": []} for name in CONFIGS
    }
    for period in list_periods():
        data = load_period(period)
        rank_panel = ranks(data["features"])
        n_val = min(800, int(0.2 * len(data["features"])))
        n_train = len(data["features"]) - n_val
        returns = data["returns"].iloc[n_train:]
        for name in CONFIGS:
            raw = factor(rank_panel, name)
            _, state = smooth(raw[:n_train], span=120)
            prediction, _ = smooth(raw[n_train:], span=120, initial=state)
            pred_frame = pd.DataFrame(prediction, index=returns.index, columns=returns.columns)
            pnl = net_returns(pred_frame, returns)
            sharpe = float(pnl.mean() / pnl.std()) if pnl.std() > 0 else 0.0
            turnover = float(pred_frame.shift(1).fillna(0.0).diff().abs().sum(axis=1).mean())
            bucket = "early" if int(period) <= 19 else "late"
            results[name][bucket].append(sharpe)
            results[name]["all"].append(sharpe)
            results[name]["turnover"].append(turnover)
        print(f"screened period {period}")
    rows = []
    for name, values in results.items():
        rows.append({
            "name": name,
            "early_mean_fold_sharpe": float(np.mean(values["early"])),
            "late_mean_fold_sharpe": float(np.mean(values["late"])),
            "all_mean_fold_sharpe": float(np.mean(values["all"])),
            "late_positive": int(np.sum(np.array(values["late"]) > 0)),
            "mean_turnover": float(np.mean(values["turnover"])),
        })
    rows.sort(key=lambda row: row["early_mean_fold_sharpe"], reverse=True)
    with open("novelty_factor_screen.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:+.5f}"))


if __name__ == "__main__":
    screen()
