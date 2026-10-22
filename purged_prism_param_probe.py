"""Fixed local parameter screen for the embargo-aware Prism."""

from __future__ import annotations

import numpy as np
import pandas as pd

from blend_purged_probe import PurgedPrism
from data_loader import list_periods, load_period
from evaluation import backtest


def pooled(parts: list[pd.Series]) -> float:
    x = pd.concat(parts, ignore_index=True)
    return float(x.mean() / x.std()) if x.std() else 0.0


def main() -> None:
    # Small grid declared before seeing the results; 001--019 is the selection block.
    configs = [
        (120, 0.75, 0.01, 10000),
        (60, 0.50, 0.01, 10000),
        (180, 0.50, 0.01, 10000),
        (240, 0.50, 0.01, 10000),
        (120, 0.50, 0.10, 10000),
        (120, 0.50, 1.00, 10000),
        (120, 0.50, 0.01, 5000),
    ]
    rows = []
    for span, strength, ridge, lookback in configs:
        early, late, all_pnl = [], [], []
        turnover = []
        for period in list_periods():
            data = load_period(period)
            n_val = min(800, int(0.2 * len(data["features"])))
            n_train = len(data["features"]) - n_val
            model = PurgedPrism()
            model.span = span
            model.regime_strength = strength
            model.ridge = ridge
            model.lookback = lookback
            model.train(data["features"].iloc[:n_train], data["target"].iloc[:n_train])
            pred = model.predict(data["features"].iloc[n_train:])
            ret = data["returns"].iloc[n_train:][pred.columns]
            pnl = backtest(pred, ret)
            all_pnl.append(pnl)
            (early if int(period) <= 19 else late).append(pnl)
            turnover.append(float(pred.shift(1).fillna(0.0).diff().abs().sum(axis=1).mean()))
        rows.append({
            "span": span,
            "regime_strength": strength,
            "ridge": ridge,
            "lookback": lookback,
            "early_net": pooled(early),
            "late_net": pooled(late),
            "full_net": pooled(all_pnl),
            "late_pos": sum(float(x.mean() / x.std()) > 0 for x in late),
            "turnover": float(np.mean(turnover)),
        })
        print(rows[-1], flush=True)
    print(pd.DataFrame(rows).sort_values("full_net", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
