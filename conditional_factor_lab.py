"""Offline-only screen of a predeclared conditional nonlinear factor family.

The supplied target is used only inside train() to fit a small ridge state
multiplier.  No external, identity, or future data is used.  The early/late
split is fixed before the results are read.
"""

import csv
import time

import numpy as np
import pandas as pd

from data_loader import list_periods, load_period
from evaluation import backtest


class ConditionalFactorLab:
    def __init__(self, kind: str):
        self.kind = kind
        self.span = 120
        self.regime_strength = 0.5
        self.ridge = 0.01
        self.lookback = 10000
        self.center = None
        self.scale = None
        self.beta = None
        self.fit_center = 0.0
        self.fit_scale = 1.0

    def _tickers(self, features: pd.DataFrame) -> list[str]:
        return sorted(features["Feature.1"].columns, key=lambda x: int(x.split(".")[-1]))

    def _factor_state(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
        tickers = self._tickers(features)
        raw, ranks = [], []
        for number in range(1, 7):
            values = features[f"Feature.{number}"][tickers].to_numpy(float)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            raw.append(values)
            order = np.argsort(np.argsort(values, axis=1, kind="mergesort"), axis=1, kind="mergesort")
            ranks.append((order.astype(float) - 9.5) / 9.5)
        core = sum(ranks[:4]) / 4.0
        tail = -core * core * core
        if self.kind == "core_only":
            factor = tail
        elif self.kind == "core_f5":
            factor = tail + core * ranks[4]
        elif self.kind == "core_f6":
            factor = tail + core * ranks[5]
        elif self.kind == "core_f6_high":
            factor = tail + 1.5 * core * ranks[5]
        elif self.kind == "core_f5_f6":
            factor = tail + core * ranks[4] + core * ranks[5]
        else:
            raise ValueError(self.kind)
        factor -= factor.mean(axis=1, keepdims=True)

        disagreement = np.mean(np.square(np.stack([item - core for item in ranks[:4]], axis=2)), axis=2)
        state_blocks = []
        for values in raw:
            compressed = np.sign(values) * np.log1p(np.abs(values))
            state_blocks.extend([
                np.mean(compressed, axis=1),
                np.std(compressed, axis=1),
                np.mean(np.abs(compressed - np.median(compressed, axis=1, keepdims=True)), axis=1),
            ])
        state_blocks.extend([
            np.std(disagreement, axis=1),
            np.mean(disagreement, axis=1),
            np.mean(core * ranks[4], axis=1),
            np.mean(core * ranks[5], axis=1),
            np.mean(ranks[4] * ranks[5], axis=1),
            np.mean(np.abs(ranks[4]), axis=1),
            np.mean(np.abs(ranks[5]), axis=1),
        ])
        return factor, np.nan_to_num(np.column_stack(state_blocks)), tickers

    def _smooth(self, values: np.ndarray) -> np.ndarray:
        alpha = 2.0 / (self.span + 1.0)
        state = np.zeros(values.shape[1], dtype=float)
        output = np.empty_like(values)
        for row in range(len(values)):
            state = alpha * values[row] + (1.0 - alpha) * state
            output[row] = state
        return output - output.mean(axis=1, keepdims=True)

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        factor, state, tickers = self._factor_state(features)
        start = max(0, len(state) - self.lookback)
        x0 = state[start:]
        self.center = np.median(x0, axis=0)
        self.scale = np.median(np.abs(x0 - self.center), axis=0) * 1.4826
        self.scale = np.where(self.scale > 1e-8, self.scale, 1.0)
        x = np.clip((x0 - self.center) / self.scale, -6.0, 6.0)
        labels = np.nan_to_num(target[tickers].to_numpy(float)[start:])
        y = np.mean(factor[start:] * labels, axis=1)
        self.beta = np.linalg.solve(x.T @ x / len(x) + self.ridge * np.eye(x.shape[1]), x.T @ (y - y.mean()) / len(x))
        fitted = x @ self.beta
        self.fit_center = float(fitted.mean())
        self.fit_scale = max(float(fitted.std()), 1e-12)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        factor, state, tickers = self._factor_state(features)
        x = np.clip((state - self.center) / self.scale, -6.0, 6.0)
        regime = (x @ self.beta - self.fit_center) / self.fit_scale
        values = self._smooth(factor * (1.0 + self.regime_strength * np.clip(regime, -2.0, 2.0))[:, None])
        return pd.DataFrame(values, index=features.index, columns=tickers).fillna(0.0)


def ratio(items: list[pd.Series]) -> float:
    joined = pd.concat(items, ignore_index=True)
    return float(joined.mean() / joined.std()) if joined.std() > 0 else 0.0


def evaluate(kind: str) -> dict[str, float | str]:
    pnl, gross, turnover, fold_ic, fold_sharpes, timings, all_values = [], [], [], [], [], [], []
    for period in list_periods():
        data = load_period(period)
        n_val = min(800, int(0.2 * len(data["features"])))
        n_train = len(data["features"]) - n_val
        model = ConditionalFactorLab(kind)
        began = time.perf_counter()
        model.train(data["features"].iloc[:n_train], data["target"].iloc[:n_train])
        train_time = time.perf_counter() - began
        began = time.perf_counter()
        pred = model.predict(data["features"].iloc[n_train:])
        predict_time = time.perf_counter() - began
        ret = data["returns"].iloc[n_train:]
        tgt = data["target"].iloc[n_train:]
        net = backtest(pred, ret)
        raw = (pred.shift(1) * ret).sum(axis=1)
        to = pred.shift(1).fillna(0.0).diff().abs().sum(axis=1)
        pnl.append(net); gross.append(raw); turnover.append(to); all_values.append(pred.values)
        fold_sharpes.append(ratio([net]))
        fold_ic.append(float(pred.corrwith(tgt, axis=1).mean()))
        timings.append((train_time, predict_time))
    # Correlation with the original factor output is a local diagnostic only;
    # no city coordinates are used for fitting or selection.
    return {
        "kind": kind,
        "early_net": ratio(pnl[:19]),
        "late_net": ratio(pnl[19:]),
        "full_net": ratio(pnl),
        "gross": ratio(gross),
        "mean_ic": float(np.mean(fold_ic)),
        "turnover": float(pd.concat(turnover, ignore_index=True).mean()),
        "positive_folds": int(np.sum(np.asarray(fold_sharpes) > 0)),
        "fold_std": float(np.std(fold_sharpes)),
        "train_s": float(np.mean([pair[0] for pair in timings])),
        "predict_s": float(np.mean([pair[1] for pair in timings])),
    }


def main() -> None:
    # Frozen small grid: primary common-feature tail, then the two anonymous
    # modifier features individually and jointly.  Selection is early_net only.
    configs = ["core_only", "core_f5", "core_f6", "core_f6_high", "core_f5_f6"]
    rows = []
    for kind in configs:
        row = evaluate(kind)
        rows.append(row)
        print(f"{kind:14s} early={row['early_net']:+.4f} late={row['late_net']:+.4f} full={row['full_net']:+.4f}", flush=True)
    with open("conditional_factor_screen.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    winner = max(rows, key=lambda row: row["early_net"])
    print("SELECTED_ON_001_019", winner["kind"])
    print("LATE_CONFIRMATION", f"{winner['late_net']:+.6f}")


if __name__ == "__main__":
    main()
