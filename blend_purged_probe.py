"""Local research: purge-aware Prism plus causal interaction blend.

This is a fixed diagnostic grid, not submission code. It never uses city data
to generate predictions and evaluates only the released walk-forward panels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphanova_causal_prism import CausalPrismPredictor
from alphanova_core_cross_orbit import CoreCrossOrbitPredictor
from data_loader import list_periods, load_period
from evaluation import backtest


class PurgedPrism(CausalPrismPredictor):
    """Same Prism feature map, but excludes the final 120 labels from fit."""

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        factor, state, tickers = self._factor_and_state(features)
        usable_end = max(1, len(state) - 120)
        start = max(0, usable_end - self.lookback)
        train_state = state[start:usable_end]
        center = np.median(train_state, axis=0)
        scale = np.median(np.abs(train_state - center), axis=0) * 1.4826
        scale = np.where(scale > 1e-8, scale, 1.0)
        standardized = np.clip((state - center) / scale, -6.0, 6.0)
        x = standardized[start:usable_end]
        labels = np.nan_to_num(
            target[tickers].to_numpy(dtype=float),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        y = np.mean(factor[start:usable_end] * labels[start:usable_end], axis=1)
        gram = x.T @ x / max(len(x), 1)
        rhs = x.T @ (y - y.mean()) / max(len(x), 1)
        self.beta = np.linalg.solve(gram + self.ridge * np.eye(x.shape[1]), rhs)
        fitted = x @ self.beta
        fit_scale = float(np.std(fitted))
        _, position_state = self._smooth(factor)
        self.tickers = tickers
        self.state_center = center
        self.state_scale = scale
        self.beta = self.beta
        self.fit_center = float(np.mean(fitted))
        self.fit_scale = fit_scale if fit_scale > 1e-12 else 1.0
        self.position_state = position_state


class Blend:
    def __init__(self, span: int, weight: float, purged: bool = True):
        self.prism = PurgedPrism() if purged else CausalPrismPredictor()
        self.cross = CoreCrossOrbitPredictor()
        self.cross.span = span
        self.weight = weight

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        self.prism.train(features, target)
        self.cross.train(features, target)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        first = self.prism.predict(features)
        second = self.cross.predict(features)[first.columns]
        raw = first.to_numpy() + self.weight * second.to_numpy()
        raw -= raw.mean(axis=1, keepdims=True)
        return pd.DataFrame(raw, index=features.index, columns=first.columns)


def pooled(parts: list[pd.Series]) -> float:
    x = pd.concat(parts, ignore_index=True)
    return float(x.mean() / x.std()) if x.std() else 0.0


def main() -> None:
    configs = [(0, 0.0, True)]
    configs += [(s, w, True) for s in (120, 240, 480) for w in (0.5, 1.0, 2.0, 4.0)]
    rows = []
    for span, weight, purged in configs:
        all_pnl, early, late, turnovers = [], [], [], []
        for period in list_periods():
            data = load_period(period)
            n_val = min(800, int(0.2 * len(data["features"])))
            n_train = len(data["features"]) - n_val
            if span == 0:
                model = PurgedPrism()
            else:
                model = Blend(span, weight, purged=purged)
            model.train(data["features"].iloc[:n_train], data["target"].iloc[:n_train])
            pred = model.predict(data["features"].iloc[n_train:])
            ret = data["returns"].iloc[n_train:][pred.columns]
            pnl = backtest(pred, ret)
            all_pnl.append(pnl)
            (early if int(period) <= 19 else late).append(pnl)
            turnovers.append(float(pred.shift(1).fillna(0.0).diff().abs().sum(axis=1).mean()))
        rows.append({
            "span": span,
            "weight": weight,
            "early_net": pooled(early),
            "late_net": pooled(late),
            "full_net": pooled(all_pnl),
            "late_pos": sum(float(x.mean() / x.std()) > 0 for x in late),
            "turnover": float(np.mean(turnovers)),
        })
        print(rows[-1], flush=True)
    print(pd.DataFrame(rows).sort_values("full_net", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
