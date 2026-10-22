"""Causal conditional-factor research, evaluated on the supplied walk-forward splits."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from data_loader import list_periods, load_period
from evaluation import backtest


def robust_standardize(values: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.clip((values - center) / scale, -6.0, 6.0)


def make_factor_and_state(
    features: pd.DataFrame,
    variant: str,
    interaction_weight: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    tickers = [f"ticker.{i}" for i in range(1, 21)]
    raw = [features[f"Feature.{i}"][tickers].to_numpy(dtype=float) for i in range(1, 7)]
    ranks = []
    for item in raw:
        ranks.append((pd.DataFrame(item).rank(axis=1).to_numpy() - 10.5) / 9.5)
    core = sum(ranks[:4]) / 4.0
    if variant == "cubic_interaction":
        factor = -np.power(core, 3) + interaction_weight * core * ranks[5]
    elif variant == "linear_interaction":
        factor = -core + interaction_weight * core * ranks[5]
    else:
        factor = -core
    disagreement = np.mean(np.square(np.stack([item - core for item in ranks[:4]], axis=2)), axis=2)

    state_blocks = []
    for item in raw:
        clipped = np.sign(item) * np.log1p(np.abs(item))
        state_blocks.extend([
            np.mean(clipped, axis=1),
            np.std(clipped, axis=1),
            np.mean(np.abs(clipped - np.median(clipped, axis=1, keepdims=True)), axis=1),
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
    state = np.column_stack(state_blocks)
    return factor, state, tickers


def smooth(values: np.ndarray, span: int, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    alpha = 2.0 / (span + 1.0)
    previous = np.zeros(values.shape[1], dtype=float) if state is None else state.copy()
    output = np.empty_like(values)
    for row in range(len(values)):
        previous = alpha * values[row] + (1.0 - alpha) * previous
        output[row] = previous
    output = output - output.mean(axis=1, keepdims=True)
    return output, previous - previous.mean()


def evaluate(
    periods: list[str], ridge: float, strength: float, span: int,
    lookback: int, variant: str, interaction_weight: float,
) -> dict:
    pnls: list[pd.Series] = []
    fold_sharpes: list[float] = []
    fold_ic: list[float] = []
    for period in periods:
        data = load_period(period)
        n_val = min(800, int(0.2 * len(data["features"])))
        n_train = len(data["features"]) - n_val
        factor, state, tickers = make_factor_and_state(
            data["features"], variant, interaction_weight,
        )
        target = data["target"][tickers].to_numpy(dtype=float)
        start = max(0, n_train - lookback) if lookback else 0
        train_state = state[start:n_train]
        center = np.median(train_state, axis=0)
        scale = np.median(np.abs(train_state - center), axis=0) * 1.4826
        scale[scale < 1e-8] = 1.0
        xs = robust_standardize(state, center, scale)
        y = np.mean(factor[start:n_train] * target[start:n_train], axis=1)
        x = xs[start:n_train]
        gram = x.T @ x / len(x)
        beta = np.linalg.solve(gram + ridge * np.eye(x.shape[1]), x.T @ (y - y.mean()) / len(x))
        fitted_train = x @ beta
        fitted_center = float(np.mean(fitted_train))
        fitted_scale = float(np.std(fitted_train))
        fitted = xs[n_train:] @ beta
        if fitted_scale > 1e-12:
            fitted = (fitted - fitted_center) / fitted_scale
        else:
            fitted = np.zeros_like(fitted)
        weights = 1.0 + strength * np.clip(fitted, -2.0, 2.0)
        raw_train = factor[:n_train]
        _, smoothing_state = smooth(raw_train, span)
        predictions, _ = smooth(factor[n_train:] * weights[:, None], span, smoothing_state)
        pred = pd.DataFrame(predictions, index=data["returns"].index[n_train:], columns=tickers)
        returns = data["returns"].iloc[n_train:][tickers]
        pnl = backtest(pred, returns)
        pnls.append(pnl)
        fold_sharpe = float(pnl.mean() / pnl.std())
        corr = float(pred.corrwith(data["target"].iloc[n_train:][tickers], axis=1).mean())
        fold_sharpes.append(fold_sharpe)
        fold_ic.append(corr)
        print(f"{period}: sharpe={fold_sharpe:+.4f} ic={corr:+.4f}")
    joined = pd.concat(pnls, ignore_index=True)
    return {
        "sharpe": float(joined.mean() / joined.std()),
        "ic": float(np.mean(fold_ic)),
        "folds": int(sum(score > 0 for score in fold_sharpes)),
        "fold_std": float(np.std(fold_sharpes)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--periods", type=int, default=12)
    parser.add_argument("--start-index", type=int, default=-1)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--strength", type=float, default=0.25)
    parser.add_argument("--span", type=int, default=50)
    parser.add_argument("--lookback", type=int, default=10000)
    parser.add_argument("--variant", choices=("core", "linear_interaction", "cubic_interaction"), default="core")
    parser.add_argument("--interaction-weight", type=float, default=1.0)
    args = parser.parse_args()
    available_periods = list_periods()
    chosen_periods = (
        available_periods[args.start_index : args.start_index + args.periods]
        if args.start_index >= 0
        else available_periods[-args.periods :]
    )
    result = evaluate(
        chosen_periods, args.ridge, args.strength, args.span,
        args.lookback, args.variant, args.interaction_weight,
    )
    print("=" * 64)
    print(
        f"sharpe={result['sharpe']:+.4f} ic={result['ic']:+.4f} "
        f"positive_folds={result['folds']}/{args.periods} fold_std={result['fold_std']:.4f}"
    )


if __name__ == "__main__":
    main()
