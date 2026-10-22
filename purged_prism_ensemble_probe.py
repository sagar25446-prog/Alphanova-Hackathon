"""Local robustness probe for fixed ensembles of purge-aware Prism variants."""

from __future__ import annotations

import numpy as np
import pandas as pd

from blend_purged_probe import PurgedPrism
from data_loader import list_periods, load_period
from evaluation import backtest


class Ensemble:
    def __init__(self, specs: tuple[tuple[int, float, float], ...]):
        self.specs = specs
        self.models = []

    def train(self, features, target):
        self.models = []
        for span, strength, ridge in self.specs:
            model = PurgedPrism()
            model.span = span
            model.regime_strength = strength
            model.ridge = ridge
            model.train(features, target)
            self.models.append(model)

    def predict(self, features):
        preds = [model.predict(features) for model in self.models]
        raw = np.mean([p.to_numpy(float) for p in preds], axis=0)
        raw -= raw.mean(axis=1, keepdims=True)
        return pd.DataFrame(raw, index=preds[0].index, columns=preds[0].columns)


def pooled(parts):
    x = pd.concat(parts, ignore_index=True)
    return float(x.mean() / x.std()) if x.std() else 0.0


def main():
    configs = {
        "120+180": ((120, 0.5, 0.01), (180, 0.5, 0.01)),
        "120+240": ((120, 0.5, 0.01), (240, 0.5, 0.01)),
        "120+ridge1": ((120, 0.5, 0.01), (120, 0.5, 1.0)),
        "120+no_regime": ((120, 0.5, 0.01), (120, 0.0, 0.01)),
    }
    rows = []
    for name, specs in configs.items():
        early, late, all_pnl, turnovers = [], [], [], []
        for period in list_periods():
            data = load_period(period)
            n_val = min(800, int(0.2 * len(data["features"])))
            n_train = len(data["features"]) - n_val
            model = Ensemble(specs)
            model.train(data["features"].iloc[:n_train], data["target"].iloc[:n_train])
            pred = model.predict(data["features"].iloc[n_train:])
            ret = data["returns"].iloc[n_train:][pred.columns]
            pnl = backtest(pred, ret)
            all_pnl.append(pnl)
            (early if int(period) <= 19 else late).append(pnl)
            turnovers.append(float(pred.shift(1).fillna(0.0).diff().abs().sum(axis=1).mean()))
        rows.append({
            "name": name,
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
