"""Offline research only: constrained causal ridge screen for an orthogonal family.

The library and selection split are fixed before execution.  All candidates use
only contemporaneous cross-sectional ranks of Features 5 and 6; they deliberately
exclude the Feature 1--4 common component used by the rejected submission.
"""

from __future__ import annotations

import csv

import numpy as np
import pandas as pd

from data_loader import list_periods, load_period


CONFIGS = (
    ("f56_ridge_001", "f56", 0.01, 120),
    ("f56_ridge_010", "f56", 0.10, 120),
    ("f56_ridge_100", "f56", 1.00, 120),
    ("f56_ridge_010_s240", "f56", 0.10, 240),
    ("f56tail_ridge_010", "f56tail", 0.10, 120),
    ("f56tail_ridge_100", "f56tail", 1.00, 120),
    ("f56spread_ridge_010", "f56spread", 0.10, 120),
    ("f56spread_ridge_100", "f56spread", 1.00, 120),
)


def rank_panels(features: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    tickers = sorted(features["Feature.1"].columns, key=lambda x: int(x.split(".")[-1]))
    blocks = []
    for number in range(1, 7):
        values = features[f"Feature.{number}"][tickers].to_numpy(dtype=float)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        ordered = pd.DataFrame(values).rank(axis=1, method="average").to_numpy(dtype=float)
        blocks.append((ordered - 10.5) / 9.5)
    return np.stack(blocks, axis=2), tickers


def design(ranks: np.ndarray, family: str) -> np.ndarray:
    f1, f2, f3, f4, f5, f6 = [ranks[:, :, number] for number in range(6)]
    c5 = f5 ** 2 - np.mean(f5 ** 2, axis=1, keepdims=True)
    c6 = f6 ** 2 - np.mean(f6 ** 2, axis=1, keepdims=True)
    common = [
        f5,
        f6,
        f5 * f6,
        c5,
        c6,
        f5 ** 3,
        f6 ** 3,
        c5 * f6,
        f5 * c6,
        f5 * (np.abs(f6) - np.mean(np.abs(f6), axis=1, keepdims=True)),
        f6 * (np.abs(f5) - np.mean(np.abs(f5), axis=1, keepdims=True)),
    ]
    if family == "f56":
        blocks = common
    elif family == "f56tail":
        blocks = common[2:]
    elif family == "f56spread":
        spread12 = f1 - f2
        spread34 = f3 - f4
        blocks = common + [
            spread12 * f5,
            spread12 * f6,
            spread34 * f5,
            spread34 * f6,
        ]
    else:
        raise ValueError(f"unknown family: {family}")
    return np.stack(blocks, axis=2)


def smooth(values: np.ndarray, span: int, initial: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    alpha = 2.0 / (span + 1.0)
    previous = np.zeros(values.shape[1], dtype=float) if initial is None else initial.copy()
    output = np.empty_like(values)
    for row in range(len(values)):
        previous = alpha * values[row] + (1.0 - alpha) * previous
        output[row] = previous
    output -= output.mean(axis=1, keepdims=True)
    return output, previous - previous.mean()


def fit_predict(x: np.ndarray, labels: np.ndarray, ridge: float, n_train: int) -> np.ndarray:
    start = max(0, n_train - 10000)
    x_train = x[start:n_train].reshape(-1, x.shape[2])
    y_train = labels[start:n_train].reshape(-1)
    good = np.isfinite(y_train) & np.isfinite(x_train).all(axis=1)
    x_train = x_train[good]
    y_train = y_train[good]
    gram = x_train.T @ x_train / max(len(x_train), 1)
    rhs = x_train.T @ (y_train - y_train.mean()) / max(len(x_train), 1)
    beta = np.linalg.solve(gram + ridge * np.eye(x.shape[2]), rhs)
    return np.einsum("tak,k->ta", x, beta)


def net_returns(predictions: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    pnl = (predictions.shift(1) * returns).sum(axis=1)
    turnover = predictions.shift(1).fillna(0.0).diff().abs().sum(axis=1)
    pnl = pnl - 0.0005 * turnover
    pnl.iloc[0] = 0.0
    return pnl


def screen() -> None:
    results: dict[str, dict[str, list[float]]] = {
        name: {"early": [], "late": [], "all": [], "turnover": []} for name, _, _, _ in CONFIGS
    }
    for period in list_periods():
        data = load_period(period)
        rank_data, tickers = rank_panels(data["features"])
        labels = data["target"][tickers].to_numpy(dtype=float)
        labels = np.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)
        n_val = min(800, int(0.2 * len(rank_data)))
        n_train = len(rank_data) - n_val
        returns = data["returns"].iloc[n_train:][tickers]
        designs = {family: design(rank_data, family) for _, family, _, _ in CONFIGS}
        for name, family, ridge, span in CONFIGS:
            raw = fit_predict(designs[family], labels, ridge, n_train)
            _, state = smooth(raw[:n_train], span)
            prediction, _ = smooth(raw[n_train:], span, state)
            pred_frame = pd.DataFrame(prediction, index=returns.index, columns=tickers)
            pnl = net_returns(pred_frame, returns)
            sharpe = float(pnl.mean() / pnl.std()) if pnl.std() > 0 else 0.0
            turnover = float(pred_frame.shift(1).fillna(0.0).diff().abs().sum(axis=1).mean())
            bucket = "early" if int(period) <= 19 else "late"
            results[name][bucket].append(sharpe)
            results[name]["all"].append(sharpe)
            results[name]["turnover"].append(turnover)
        print(f"screened period {period}", flush=True)
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
    with open("novelty_ridge_screen.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:+.5f}"))


if __name__ == "__main__":
    screen()
