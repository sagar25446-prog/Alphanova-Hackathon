"""Causal temporal-feature audit using the organizer's net-PnL evaluator."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from data_loader import list_periods, load_period
from evaluation import backtest


def xs_rank(frame: pd.DataFrame) -> np.ndarray:
    n = frame.shape[1]
    return (frame.rank(axis=1).to_numpy(dtype=float) - (n + 1.0) / 2.0) / ((n - 1.0) / 2.0)


def xs_demean(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=1, keepdims=True)


def ewm(values: np.ndarray, span: int) -> np.ndarray:
    return pd.DataFrame(values).ewm(span=span, adjust=False).mean().to_numpy()


def position_smooth(values: np.ndarray, span: int) -> np.ndarray:
    return xs_demean(ewm(xs_demean(values), span))


def formulas(features: pd.DataFrame, selected: set[str] | None = None) -> dict[str, np.ndarray]:
    tickers = [f"ticker.{i}" for i in range(1, 21)]
    r = [xs_rank(features[f"Feature.{i}"][tickers]) for i in range(1, 7)]
    bases = {
        "core": xs_demean(sum(r[:4]) / 4.0),
        "f5": xs_demean(r[4]),
        "f6": xs_demean(r[5]),
        "disagreement": xs_demean(np.mean(np.square(np.stack([x - sum(r[:4]) / 4.0 for x in r[:4]], axis=2)), axis=2)),
    }
    out: dict[str, np.ndarray] = {}
    for base_name, base in bases.items():
        neg_level = f"-{base_name}"
        pos_level = f"+{base_name}"
        if selected is None or neg_level in selected:
            out[neg_level] = -base
        if selected is None or pos_level in selected:
            out[pos_level] = base
        for lag in (1, 3, 6, 12, 24, 48, 96, 168):
            neg_name = f"-{base_name}_delta_{lag}"
            pos_name = f"+{base_name}_delta_{lag}"
            if selected is not None and neg_name not in selected and pos_name not in selected:
                continue
            previous = np.vstack((np.zeros((lag, base.shape[1])), base[:-lag]))
            if selected is None or neg_name in selected:
                out[neg_name] = xs_demean(previous - base)
            if selected is None or pos_name in selected:
                out[pos_name] = xs_demean(base - previous)
        for fast, slow in ((3, 12), (6, 24), (12, 48), (24, 96), (48, 168)):
            neg_name = f"-{base_name}_macd_{fast}_{slow}"
            pos_name = f"+{base_name}_macd_{fast}_{slow}"
            if selected is not None and neg_name not in selected and pos_name not in selected:
                continue
            macd = xs_demean(ewm(base, fast) - ewm(base, slow))
            if selected is None or neg_name in selected:
                out[neg_name] = -macd
            if selected is None or pos_name in selected:
                out[pos_name] = macd
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--span", type=int, default=50)
    parser.add_argument("--periods", type=int, default=31)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--select", type=str, default="")
    args = parser.parse_args()
    periods = list_periods()[-args.periods :]
    selected = {item.strip() for item in args.select.split(",") if item.strip()}
    pnl: dict[str, list[pd.Series]] = {}
    ics: dict[str, list[float]] = {}
    folds: dict[str, list[float]] = {}
    for period in periods:
        data = load_period(period)
        n_val = min(800, int(0.2 * len(data["features"])))
        tickers = [f"ticker.{i}" for i in range(1, 21)]
        returns = data["returns"].iloc[-n_val:][tickers]
        target = data["target"].iloc[-n_val:][tickers]
        for name, raw in formulas(data["features"], selected or None).items():
            values = position_smooth(raw, args.span)[-n_val:]
            pred = pd.DataFrame(values, index=returns.index, columns=tickers)
            item = backtest(pred, returns)
            pnl.setdefault(name, []).append(item)
            ics.setdefault(name, []).append(float(pred.corrwith(target, axis=1).mean()))
            folds.setdefault(name, []).append(float(item.mean() / item.std()))
        print(f"{period}: completed")
    rows = []
    for name, pieces in pnl.items():
        joined = pd.concat(pieces, ignore_index=True)
        rows.append((float(joined.mean() / joined.std()), float(np.mean(ics[name])), sum(x > 0 for x in folds[name]), float(np.std(folds[name])), name))
    for sharpe, ic, count, spread, name in sorted(rows, reverse=True)[: args.top]:
        print(f"{name:32s} sharpe={sharpe:+.4f} ic={ic:+.4f} folds={count:2d}/{len(periods)} sd={spread:.4f}")


if __name__ == "__main__":
    main()
