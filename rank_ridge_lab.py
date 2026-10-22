"""Offline research only: a predeclared low-capacity cross-sectional ridge screen.

This file is not a competition submission.  It deliberately uses only the
provided local feature/target/return panels.  Model selection is restricted to
the first 19 walk-forward periods; periods 020--031 are left untouched until
the predeclared winner is selected.
"""

import csv
import time

import numpy as np
import pandas as pd

from data_loader import list_periods, load_period
from evaluation import backtest


class RankRidgeLab:
    """Stateless pooled ridge on bounded, contemporaneous rank transformations."""

    def __init__(self, basis: str, ridge: float):
        self.basis = basis
        self.ridge = ridge
        self.beta = None

    def _tickers(self, features: pd.DataFrame) -> list[str]:
        return sorted(features["Feature.1"].columns, key=lambda x: int(x.split(".")[-1]))

    def _terms(self, features: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        tickers = self._tickers(features)
        panels = []
        for number in range(1, 7):
            values = features[f"Feature.{number}"][tickers].to_numpy(float)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            order = np.argsort(np.argsort(values, axis=1, kind="mergesort"), axis=1, kind="mergesort")
            panels.append((order.astype(float) - 9.5) / 9.5)
        rank = np.stack(panels, axis=2)
        blocks = [rank]
        if "quadratic" in self.basis:
            square = rank * rank
            blocks.append(square - square.mean(axis=1, keepdims=True))
        if "cubic" in self.basis:
            blocks.append(rank * rank * rank)
        if "pairs" in self.basis:
            pair_blocks = []
            for left in range(6):
                for right in range(left + 1, 6):
                    pair = rank[:, :, left] * rank[:, :, right]
                    pair_blocks.append(pair - pair.mean(axis=1, keepdims=True))
            blocks.append(np.stack(pair_blocks, axis=2))
        return np.concatenate(blocks, axis=2), tickers

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        terms, tickers = self._terms(features)
        # A fixed recent-history cap makes all variants comparable and avoids
        # giving old, independently obfuscated periods disproportionate weight.
        terms = terms[-10000:]
        labels = target[tickers].to_numpy(float)[-10000:]
        x = terms.reshape(-1, terms.shape[2])
        y = np.nan_to_num(labels.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
        gram = (x.T @ x) / len(x)
        rhs = (x.T @ y) / len(x)
        self.beta = np.linalg.solve(gram + self.ridge * np.eye(gram.shape[0]), rhs)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        terms, tickers = self._terms(features)
        values = terms @ self.beta
        values = values - values.mean(axis=1, keepdims=True)
        return pd.DataFrame(values, index=features.index, columns=tickers).fillna(0.0)


def sharpe(values: list[pd.Series]) -> float:
    series = pd.concat(values, ignore_index=True)
    return float(series.mean() / series.std()) if series.std() > 0 else 0.0


def evaluate(basis: str, ridge: float) -> dict[str, float | str]:
    fold_pnl = []
    fold_gross = []
    fold_turnover = []
    fold_ic = []
    train_seconds = []
    predict_seconds = []
    for period in list_periods():
        data = load_period(period)
        n_val = min(800, int(0.2 * len(data["features"])))
        n_train = len(data["features"]) - n_val
        model = RankRidgeLab(basis, ridge)
        started = time.perf_counter()
        model.train(data["features"].iloc[:n_train], data["target"].iloc[:n_train])
        train_seconds.append(time.perf_counter() - started)
        started = time.perf_counter()
        prediction = model.predict(data["features"].iloc[n_train:])
        predict_seconds.append(time.perf_counter() - started)
        returns = data["returns"].iloc[n_train:]
        target = data["target"].iloc[n_train:]
        pnl = backtest(prediction, returns)
        gross = (prediction.shift(1) * returns).sum(axis=1)
        turnover = prediction.shift(1).fillna(0.0).diff().abs().sum(axis=1)
        fold_pnl.append(pnl)
        fold_gross.append(gross)
        fold_turnover.append(turnover)
        fold_ic.append(float(prediction.corrwith(target, axis=1).mean()))
    early_pnl = fold_pnl[:19]
    late_pnl = fold_pnl[19:]
    fold_sharpes = np.array([sharpe([item]) for item in fold_pnl])
    return {
        "basis": basis,
        "ridge": ridge,
        "early_net": sharpe(early_pnl),
        "late_net": sharpe(late_pnl),
        "full_net": sharpe(fold_pnl),
        "gross": sharpe(fold_gross),
        "mean_ic": float(np.mean(fold_ic)),
        "turnover": float(pd.concat(fold_turnover, ignore_index=True).mean()),
        "positive_folds": int(np.sum(fold_sharpes > 0)),
        "fold_std": float(fold_sharpes.std()),
        "train_s": float(np.mean(train_seconds)),
        "predict_s": float(np.mean(predict_seconds)),
    }


def main() -> None:
    # Fixed before looking at the late-period scores: four basis choices x two
    # L2 strengths.  The selection rule is highest early_net, with late_net
    # reported only after that selection.
    grid = [
        ("linear", 0.1),
        ("linear", 1.0),
        ("linear_cubic", 0.1),
        ("linear_cubic", 1.0),
        ("linear_quadratic_cubic", 0.1),
        ("linear_quadratic_cubic", 1.0),
        ("linear_cubic_pairs", 0.1),
        ("linear_cubic_pairs", 1.0),
    ]
    results = []
    for basis, ridge in grid:
        result = evaluate(basis, ridge)
        results.append(result)
        print(
            f"{basis:25s} ridge={ridge:<3.1f} "
            f"early={result['early_net']:+.4f} late={result['late_net']:+.4f} "
            f"full={result['full_net']:+.4f} turnover={result['turnover']:.4f}"
        )
    fields = list(results[0])
    with open("rank_ridge_screen.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    winner = max(results, key=lambda row: row["early_net"])
    print("SELECTED_ON_001_019", winner["basis"], winner["ridge"])
    print("LATE_CONFIRMATION", f"{winner['late_net']:+.6f}")


if __name__ == "__main__":
    main()
