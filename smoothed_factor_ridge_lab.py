"""Offline-only, predeclared ridge blend of causal nonlinear factor panels."""

import csv
import time

import numpy as np
import pandas as pd

from data_loader import list_periods, load_period
from evaluation import backtest


class SmoothedFactorRidgeLab:
    def __init__(self, ridge: float):
        self.ridge = ridge
        self.span = 120
        self.center = None
        self.scale = None
        self.beta = None

    def _tickers(self, features: pd.DataFrame) -> list[str]:
        return sorted(features["Feature.1"].columns, key=lambda name: int(name.split(".")[-1]))

    def _factors(self, features: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        tickers = self._tickers(features)
        rank = []
        for number in range(1, 7):
            values = np.nan_to_num(features[f"Feature.{number}"][tickers].to_numpy(float))
            order = np.argsort(np.argsort(values, axis=1, kind="mergesort"), axis=1, kind="mergesort")
            rank.append((order.astype(float) - 9.5) / 9.5)
        f1, f2, f3, f4, f5, f6 = rank
        core = (f1 + f2 + f3 + f4) / 4.0
        core_sq = core * core - np.mean(core * core, axis=1, keepdims=True)
        pieces = [
            -core * core * core,
            core * f5,
            core * f6,
            core_sq * f5,
            core_sq * f6,
            core * core * core * f5,
            core * core * core * f6,
            f5 * f5 * f5 - f6 * f6 * f6,
        ]
        raw = np.stack(pieces, axis=2)
        raw = raw - raw.mean(axis=1, keepdims=True)
        alpha = 2.0 / (self.span + 1.0)
        state = np.zeros((raw.shape[1], raw.shape[2]), dtype=float)
        smoothed = np.empty_like(raw)
        for row in range(len(raw)):
            state = alpha * raw[row] + (1.0 - alpha) * state
            smoothed[row] = state
        return smoothed - smoothed.mean(axis=1, keepdims=True), tickers

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        factors, tickers = self._factors(features)
        factors = factors[-10000:]
        labels = np.nan_to_num(target[tickers].to_numpy(float)[-10000:])
        x = factors.reshape(-1, factors.shape[2])
        y = labels.reshape(-1)
        self.center = x.mean(axis=0)
        self.scale = np.maximum(x.std(axis=0), 1e-8)
        standardized = (x - self.center) / self.scale
        gram = standardized.T @ standardized / len(standardized)
        rhs = standardized.T @ y / len(standardized)
        self.beta = np.linalg.solve(gram + self.ridge * np.eye(gram.shape[0]), rhs)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        factors, tickers = self._factors(features)
        x = factors.reshape(-1, factors.shape[2])
        values = ((x - self.center) / self.scale) @ self.beta
        values = values.reshape(factors.shape[:2])
        values = values - values.mean(axis=1, keepdims=True)
        return pd.DataFrame(values, index=features.index, columns=tickers).fillna(0.0)


def ratio(parts: list[pd.Series]) -> float:
    joined = pd.concat(parts, ignore_index=True)
    return float(joined.mean() / joined.std()) if joined.std() else 0.0


def evaluate(ridge: float) -> dict[str, float]:
    pnl, gross, turnover, ic, fold, timings = [], [], [], [], [], []
    for period in list_periods():
        data = load_period(period)
        n_val = min(800, int(0.2 * len(data["features"])))
        n_train = len(data["features"]) - n_val
        model = SmoothedFactorRidgeLab(ridge)
        started = time.perf_counter(); model.train(data["features"].iloc[:n_train], data["target"].iloc[:n_train]); train_t = time.perf_counter() - started
        started = time.perf_counter(); pred = model.predict(data["features"].iloc[n_train:]); pred_t = time.perf_counter() - started
        ret, tgt = data["returns"].iloc[n_train:], data["target"].iloc[n_train:]
        net = backtest(pred, ret)
        pnl.append(net); gross.append((pred.shift(1) * ret).sum(axis=1))
        turnover.append(pred.shift(1).fillna(0.0).diff().abs().sum(axis=1))
        ic.append(float(pred.corrwith(tgt, axis=1).mean())); fold.append(ratio([net])); timings.append((train_t, pred_t))
    return {
        "ridge": ridge,
        "early_net": ratio(pnl[:19]), "late_net": ratio(pnl[19:]), "full_net": ratio(pnl),
        "gross": ratio(gross), "mean_ic": float(np.mean(ic)),
        "turnover": float(pd.concat(turnover, ignore_index=True).mean()),
        "positive_folds": int(np.sum(np.asarray(fold) > 0)), "fold_std": float(np.std(fold)),
        "train_s": float(np.mean([item[0] for item in timings])),
        "predict_s": float(np.mean([item[1] for item in timings])),
    }


def main() -> None:
    # Frozen L2 grid; select on 001--019 only and report the one late confirmation.
    rows = [evaluate(ridge) for ridge in (0.1, 1.0, 10.0)]
    for row in rows:
        print(f"ridge={row['ridge']:>4.1f} early={row['early_net']:+.4f} late={row['late_net']:+.4f} full={row['full_net']:+.4f}", flush=True)
    with open("smoothed_factor_ridge_screen.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    winner = max(rows, key=lambda row: row["early_net"])
    print("SELECTED_ON_001_019", winner["ridge"])
    print("LATE_CONFIRMATION", f"{winner['late_net']:+.6f}")


if __name__ == "__main__":
    main()
