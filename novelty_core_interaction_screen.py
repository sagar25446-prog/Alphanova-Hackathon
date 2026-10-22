"""Offline research only: fixed high-novelty core--Feature-6 interaction grid."""

from __future__ import annotations

import csv

import numpy as np
import pandas as pd

from data_loader import list_periods, load_period


WEIGHTS = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0)


def rank_panels(features: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    tickers = sorted(features["Feature.1"].columns, key=lambda x: int(x.split(".")[-1]))
    blocks = []
    for number in range(1, 7):
        values = features[f"Feature.{number}"][tickers].to_numpy(dtype=float)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        ordered = pd.DataFrame(values).rank(axis=1, method="average").to_numpy(dtype=float)
        blocks.append((ordered - 10.5) / 9.5)
    return np.stack(blocks, axis=2), tickers


def smooth(values: np.ndarray, initial: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    alpha = 2.0 / 121.0
    previous = np.zeros(values.shape[1], dtype=float) if initial is None else initial.copy()
    output = np.empty_like(values)
    for row in range(len(values)):
        previous = alpha * values[row] + (1.0 - alpha) * previous
        output[row] = previous
    output -= output.mean(axis=1, keepdims=True)
    return output, previous - previous.mean()


def net_returns(predictions: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    pnl = (predictions.shift(1) * returns).sum(axis=1)
    turnover = predictions.shift(1).fillna(0.0).diff().abs().sum(axis=1)
    pnl = pnl - 0.0005 * turnover
    pnl.iloc[0] = 0.0
    return pnl


def screen() -> None:
    results: dict[float, dict[str, list[float]]] = {
        weight: {"early": [], "late": [], "all": [], "turnover": []} for weight in WEIGHTS
    }
    for period in list_periods():
        data = load_period(period)
        ranks, tickers = rank_panels(data["features"])
        core = ranks[:, :, :4].mean(axis=2)
        sixth = ranks[:, :, 5]
        n_val = min(800, int(0.2 * len(ranks)))
        n_train = len(ranks) - n_val
        returns = data["returns"].iloc[n_train:][tickers]
        for weight in WEIGHTS:
            raw = -(core + weight * core * sixth)
            raw -= raw.mean(axis=1, keepdims=True)
            _, state = smooth(raw[:n_train])
            prediction, _ = smooth(raw[n_train:], state)
            pred_frame = pd.DataFrame(prediction, index=returns.index, columns=tickers)
            pnl = net_returns(pred_frame, returns)
            sharpe = float(pnl.mean() / pnl.std()) if pnl.std() > 0 else 0.0
            turnover = float(pred_frame.shift(1).fillna(0.0).diff().abs().sum(axis=1).mean())
            bucket = "early" if int(period) <= 19 else "late"
            results[weight][bucket].append(sharpe)
            results[weight]["all"].append(sharpe)
            results[weight]["turnover"].append(turnover)
        print(f"screened period {period}", flush=True)
    rows = []
    for weight, values in results.items():
        rows.append({
            "interaction_weight": weight,
            "early_mean_fold_sharpe": float(np.mean(values["early"])),
            "late_mean_fold_sharpe": float(np.mean(values["late"])),
            "all_mean_fold_sharpe": float(np.mean(values["all"])),
            "late_positive": int(np.sum(np.array(values["late"]) > 0)),
            "mean_turnover": float(np.mean(values["turnover"])),
        })
    rows.sort(key=lambda row: row["early_mean_fold_sharpe"], reverse=True)
    with open("novelty_core_interaction_inverse_screen.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:+.5f}"))


if __name__ == "__main__":
    screen()
