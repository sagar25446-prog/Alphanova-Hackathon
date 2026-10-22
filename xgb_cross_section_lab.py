"""Offline-only low-capacity XGBoost screen for a causal cross-sectional model.

Four fixed feature designs are tested.  The early block (001--019) selects the
winner; its late block (020--031) is reported once as confirmation.  This is
research, not a submission.
"""

import csv
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from data_loader import list_periods, load_period
from evaluation import backtest


class CrossSectionLab:
    def __init__(self, design_name: str):
        self.design_name = design_name
        self.model = None
        self.span = 120
        self.max_rows = 240000

    def _tickers(self, features: pd.DataFrame) -> list[str]:
        return sorted(features["Feature.1"].columns, key=lambda value: int(value.split(".")[-1]))

    def _rank(self, values: np.ndarray) -> np.ndarray:
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        ordered = np.argsort(np.argsort(values, axis=1, kind="mergesort"), axis=1, kind="mergesort")
        return (ordered.astype(float) - 9.5) / 9.5

    def _design(self, features: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        tickers = self._tickers(features)
        raw = [features[f"Feature.{number}"][tickers].to_numpy(float) for number in range(1, 7)]
        ranks = [self._rank(values) for values in raw]
        f1, f2, f3, f4, f5, f6 = ranks
        core = (f1 + f2 + f3 + f4) / 4.0
        disagreement = np.mean(np.square(np.stack([f1 - core, f2 - core, f3 - core, f4 - core], axis=2)), axis=2)
        if self.design_name == "six_rank":
            blocks = ranks
        elif self.design_name == "difference_rank":
            blocks = [f6, f1, self._rank(raw[0] - raw[1]), self._rank(raw[2] - raw[3]), self._rank(raw[0] - raw[2]), f5]
        elif self.design_name == "symmetric_compact":
            blocks = [core, f5, f6, disagreement, core * core, f5 * f5, f6 * f6, core * f5, core * f6, f5 * f6]
        elif self.design_name == "symmetric_extended":
            blocks = [*ranks, core, disagreement, core * core, core * core * core, f5 * f5, f6 * f6, core * f5, core * f6, f5 * f6, core * core * f5, core * core * f6]
        else:
            raise ValueError(self.design_name)
        array = np.stack(blocks, axis=2)
        array -= array.mean(axis=1, keepdims=True)
        return array, tickers

    def _smooth(self, values: np.ndarray, initial: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        alpha = 2.0 / (self.span + 1.0)
        previous = np.zeros(values.shape[1], dtype=float) if initial is None else initial.copy()
        output = np.empty_like(values)
        for row in range(len(values)):
            previous = alpha * values[row] + (1.0 - alpha) * previous
            output[row] = previous
        return output - output.mean(axis=1, keepdims=True), previous - previous.mean()

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        values, tickers = self._design(features)
        labels = np.nan_to_num(target[tickers].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
        values = values[-self.max_rows // 20:]
        labels = labels[-len(values):]
        x = values.reshape(-1, values.shape[2])
        y = labels.reshape(-1)
        # Deterministic row thinning keeps repeated gate retraining cheap while
        # preserving all cross-sectional observations at sampled timestamps.
        if len(x) > self.max_rows:
            keep = np.arange(len(x)) % 2 == 0
            x, y = x[keep], y[keep]
        self.model = xgb.train(
            {
                "objective": "reg:squarederror",
                "max_depth": 2,
                "eta": 0.05,
                "subsample": 0.75,
                "colsample_bytree": 0.80,
                "min_child_weight": 1000.0,
                "reg_lambda": 20.0,
                "tree_method": "hist",
                "nthread": 4,
                "seed": 20260827,
                "verbosity": 0,
            },
            xgb.DMatrix(np.nan_to_num(x), label=y),
            num_boost_round=30,
        )
        train_raw = self.model.predict(xgb.DMatrix(np.nan_to_num(values.reshape(-1, values.shape[2])))).reshape(values.shape[:2])
        _, self.state = self._smooth(train_raw)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        values, tickers = self._design(features)
        raw = self.model.predict(xgb.DMatrix(np.nan_to_num(values.reshape(-1, values.shape[2])))).reshape(values.shape[:2])
        result, _ = self._smooth(raw, self.state)
        return pd.DataFrame(result, index=features.index, columns=tickers).fillna(0.0)


def ratio(parts: list[pd.Series]) -> float:
    joined = pd.concat(parts, ignore_index=True)
    return float(joined.mean() / joined.std()) if joined.std() else 0.0


def evaluate(design_name: str) -> dict[str, float | str]:
    net, gross, turnover, folds, ics, train_time, predict_time = [], [], [], [], [], [], []
    for period in list_periods():
        data = load_period(period)
        n_val = min(800, int(0.2 * len(data["features"])))
        n_train = len(data["features"]) - n_val
        model = CrossSectionLab(design_name)
        started = time.perf_counter(); model.train(data["features"].iloc[:n_train], data["target"].iloc[:n_train]); train_time.append(time.perf_counter() - started)
        started = time.perf_counter(); pred = model.predict(data["features"].iloc[n_train:]); predict_time.append(time.perf_counter() - started)
        ret, tgt = data["returns"].iloc[n_train:], data["target"].iloc[n_train:]
        pnl = backtest(pred, ret)
        net.append(pnl); gross.append((pred.shift(1) * ret).sum(axis=1)); turnover.append(pred.shift(1).fillna(0.0).diff().abs().sum(axis=1))
        folds.append(ratio([pnl])); ics.append(float(pred.corrwith(tgt, axis=1).mean()))
    return {
        "design": design_name,
        "early_net": ratio(net[:19]), "late_net": ratio(net[19:]), "full_net": ratio(net),
        "gross": ratio(gross), "mean_ic": float(np.mean(ics)),
        "turnover": float(pd.concat(turnover, ignore_index=True).mean()),
        "positive_folds": int(np.sum(np.asarray(folds) > 0)), "fold_std": float(np.std(folds)),
        "train_s": float(np.mean(train_time)), "predict_s": float(np.mean(predict_time)),
    }


def main() -> None:
    designs = ["six_rank", "difference_rank", "symmetric_compact", "symmetric_extended"]
    rows = []
    for design in designs:
        row = evaluate(design); rows.append(row)
        print(f"{design:20s} early={row['early_net']:+.4f} late={row['late_net']:+.4f} full={row['full_net']:+.4f} train={row['train_s']:.2f}s", flush=True)
    with open("xgb_cross_section_screen.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    winner = max(rows, key=lambda row: row["early_net"])
    print("SELECTED_ON_001_019", winner["design"])
    print("LATE_CONFIRMATION", f"{winner['late_net']:+.6f}")


if __name__ == "__main__":
    main()
