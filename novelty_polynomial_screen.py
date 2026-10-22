"""Offline research only: bounded polynomial cross-sectional ridge candidates.

Model selection is fixed to the early 001--019 block.  Periods 020--031 are
read only once as an out-of-sample confirmation block.
"""

from __future__ import annotations

import csv

import numpy as np
import pandas as pd

from data_loader import list_periods, load_period


CONFIGS = (
    ("poly2_r001", "poly2", 0.01, False),
    ("poly2_r010", "poly2", 0.10, False),
    ("poly2_r100", "poly2", 1.00, False),
    ("poly3_r010", "poly3", 0.10, False),
    ("poly3_r100", "poly3", 1.00, False),
    ("poly3_r1000", "poly3", 10.00, False),
    ("poly3_resid_r010", "poly3", 0.10, True),
    ("poly3_resid_r100", "poly3", 1.00, True),
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
    blocks = [ranks[:, :, number] for number in range(6)]
    blocks.extend(
        ranks[:, :, number] ** 2 - np.mean(ranks[:, :, number] ** 2, axis=1, keepdims=True)
        for number in range(6)
    )
    blocks.extend(ranks[:, :, number] ** 3 for number in range(6))
    for first in range(6):
        for second in range(first + 1, 6):
            blocks.append(ranks[:, :, first] * ranks[:, :, second])
    if family == "poly3":
        for first in range(6):
            for second in range(6):
                if first != second:
                    blocks.append(ranks[:, :, first] ** 2 * ranks[:, :, second])
        for first in range(6):
            for second in range(first + 1, 6):
                for third in range(second + 1, 6):
                    blocks.append(ranks[:, :, first] * ranks[:, :, second] * ranks[:, :, third])
    return np.stack(blocks, axis=2)


def core_direction(ranks: np.ndarray) -> np.ndarray:
    core = ranks[:, :, :4].mean(axis=2)
    values = -core ** 3 + core * ranks[:, :, 5]
    return values - values.mean(axis=1, keepdims=True)


def fit_predict(x: np.ndarray, labels: np.ndarray, ridge: float, n_train: int) -> np.ndarray:
    start = max(0, n_train - 10000)
    x_train = x[start:n_train].reshape(-1, x.shape[2])
    y_train = labels[start:n_train].reshape(-1)
    good = np.isfinite(y_train) & np.isfinite(x_train).all(axis=1)
    x_train = x_train[good]
    y_train = y_train[good]
    center = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    z_train = (x_train - center) / scale
    gram = z_train.T @ z_train / max(len(z_train), 1)
    rhs = z_train.T @ (y_train - y_train.mean()) / max(len(z_train), 1)
    beta = np.linalg.solve(gram + ridge * np.eye(z_train.shape[1]), rhs)
    return np.einsum("tak,k->ta", (x - center) / scale, beta)


def residualize(values: np.ndarray, direction: np.ndarray) -> np.ndarray:
    denominator = np.sum(direction * direction, axis=1, keepdims=True)
    coefficient = np.sum(values * direction, axis=1, keepdims=True) / np.maximum(denominator, 1e-12)
    result = values - coefficient * direction
    return result - result.mean(axis=1, keepdims=True)


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
    results: dict[str, dict[str, list[float]]] = {
        name: {"early": [], "late": [], "all": [], "turnover": []} for name, _, _, _ in CONFIGS
    }
    for period in list_periods():
        data = load_period(period)
        rank_data, tickers = rank_panels(data["features"])
        labels = np.nan_to_num(data["target"][tickers].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        n_val = min(800, int(0.2 * len(rank_data)))
        n_train = len(rank_data) - n_val
        returns = data["returns"].iloc[n_train:][tickers]
        designs = {family: design(rank_data, family) for _, family, _, _ in CONFIGS}
        base = core_direction(rank_data)
        for name, family, ridge, use_residual in CONFIGS:
            raw = fit_predict(designs[family], labels, ridge, n_train)
            if use_residual:
                raw = residualize(raw, base)
            _, state = smooth(raw[:n_train])
            prediction, _ = smooth(raw[n_train:], state)
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
    with open("novelty_polynomial_screen.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:+.5f}"))


if __name__ == "__main__":
    screen()
