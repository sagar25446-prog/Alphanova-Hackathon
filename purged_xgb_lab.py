"""Embargo-aware offline screen for the AlphaNova supplied data.

The public discussion established that the released target is forward-looking
at roughly 120 rows.  This lab therefore removes the last 120 labelled rows
of every training prefix before fitting.  It is research only: no server data,
city coordinate, ticker identity, or future feature is used.
"""

from __future__ import annotations

import csv
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from data_loader import list_periods, load_period
from evaluation import backtest


CONFIGS = (
    ("difference_12k", "difference", 12000),
    ("difference_5k", "difference", 5000),
    ("compact_12k", "compact", 12000),
    ("sixrank_12k", "sixrank", 12000),
)
EMBARGO = 120


class PurgedXGBLab:
    def __init__(self, design_name: str, max_rows: int):
        self.design_name = design_name
        self.max_rows = max_rows
        self.span = 120
        self.model = None
        self.position_state = None

    def _tickers(self, features: pd.DataFrame) -> list[str]:
        return sorted(features["Feature.1"].columns, key=lambda name: int(name.split(".")[-1]))

    def _rank(self, values: np.ndarray) -> np.ndarray:
        clean = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        order = np.argsort(np.argsort(clean, axis=1, kind="mergesort"), axis=1, kind="mergesort")
        return (order.astype(float) - 9.5) / 9.5

    def _design(self, features: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        tickers = self._tickers(features)
        raw = [features[f"Feature.{number}"][tickers].to_numpy(float) for number in range(1, 7)]
        ranks = [self._rank(values) for values in raw]
        f1, f2, f3, f4, f5, f6 = ranks
        core = (f1 + f2 + f3 + f4) / 4.0
        disagreement = np.mean(np.square(np.stack([f1-core, f2-core, f3-core, f4-core], axis=2)), axis=2)
        if self.design_name == "difference":
            blocks = [f6, f1, self._rank(raw[0] - raw[1]), self._rank(raw[2] - raw[3]), self._rank(raw[0] - raw[2]), f5]
        elif self.design_name == "compact":
            blocks = [core, f5, f6, disagreement, core*core, f5*f5, f6*f6, core*f5, core*f6, f5*f6]
        else:
            blocks = ranks
        result = np.stack(blocks, axis=2)
        result -= result.mean(axis=1, keepdims=True)
        return np.nan_to_num(result), tickers

    def _smooth(self, values: np.ndarray, initial: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        alpha = 2.0 / (self.span + 1.0)
        state = np.zeros(values.shape[1], dtype=float) if initial is None else initial.copy()
        output = np.empty_like(values)
        for row in range(len(values)):
            state = alpha * values[row] + (1.0-alpha) * state
            output[row] = state
        output -= output.mean(axis=1, keepdims=True)
        return output, state - state.mean()

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        design, tickers = self._design(features)
        labels = np.nan_to_num(target[tickers].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
        usable_end = max(1, len(design) - EMBARGO)
        start = max(0, usable_end - self.max_rows)
        x = design[start:usable_end].reshape(-1, design.shape[2])
        y = labels[start:usable_end].reshape(-1)
        self.model = xgb.train(
            {
                "objective": "reg:squarederror", "max_depth": 2,
                "eta": 0.05, "subsample": 0.75, "colsample_bytree": 0.80,
                "min_child_weight": 1000.0, "reg_lambda": 20.0,
                "tree_method": "hist", "nthread": 4, "seed": 20260827,
                "verbosity": 0,
            },
            xgb.DMatrix(x, label=y),
            num_boost_round=30,
        )
        fitted = self.model.predict(xgb.DMatrix(design.reshape(-1, design.shape[2]))).reshape(design.shape[:2])
        _, self.position_state = self._smooth(fitted)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        design, tickers = self._design(features)
        raw = self.model.predict(xgb.DMatrix(design.reshape(-1, design.shape[2]))).reshape(design.shape[:2])
        values, _ = self._smooth(raw, self.position_state)
        result = pd.DataFrame(values, index=features.index, columns=tickers)
        return result.sub(result.mean(axis=1), axis=0).fillna(0.0)


def ratio(parts: list[pd.Series]) -> float:
    joined = pd.concat(parts, ignore_index=True)
    return float(joined.mean() / joined.std()) if joined.std() else 0.0


def evaluate(name: str, design: str, max_rows: int) -> dict[str, float | str]:
    pnls, gross, turns, sharpe, ics, timings = [], [], [], [], [], []
    for period in list_periods():
        data = load_period(period)
        n_val = min(800, int(0.2 * len(data["features"])))
        n_train = len(data["features"]) - n_val
        model = PurgedXGBLab(design, max_rows)
        began = time.perf_counter()
        model.train(data["features"].iloc[:n_train], data["target"].iloc[:n_train])
        train_seconds = time.perf_counter() - began
        began = time.perf_counter()
        pred = model.predict(data["features"].iloc[n_train:])
        predict_seconds = time.perf_counter() - began
        returns = data["returns"].iloc[n_train:][pred.columns]
        target = data["target"].iloc[n_train:][pred.columns]
        pnl = backtest(pred, returns)
        pnls.append(pnl)
        gross.append((pred.shift(1) * returns).sum(axis=1))
        turns.append(pred.shift(1).fillna(0.0).diff().abs().sum(axis=1))
        sharpe.append(ratio([pnl]))
        ics.append(float(pred.corrwith(target, axis=1).mean()))
        timings.append((train_seconds, predict_seconds))
    return {
        "name": name,
        "early_net": ratio(pnls[:19]), "late_net": ratio(pnls[19:]), "full_net": ratio(pnls),
        "gross": ratio(gross), "mean_ic": float(np.mean(ics)),
        "turnover": float(pd.concat(turns, ignore_index=True).mean()),
        "positive_folds": int(np.sum(np.asarray(sharpe) > 0)), "fold_std": float(np.std(sharpe)),
        "train_s": float(np.mean([v[0] for v in timings])), "predict_s": float(np.mean([v[1] for v in timings])),
    }


def main() -> None:
    rows = []
    for name, design, max_rows in CONFIGS:
        row = evaluate(name, design, max_rows)
        rows.append(row)
        print(f"{name:16s} early={row['early_net']:+.4f} late={row['late_net']:+.4f} full={row['full_net']:+.4f}", flush=True)
    with open("purged_xgb_screen.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    winner = max(rows, key=lambda row: row["early_net"])
    print("SELECTED_ON_001_019", winner["name"])
    print("LATE_CONFIRMATION", f"{winner['late_net']:+.6f}")


if __name__ == "__main__":
    main()
