"""Local research harness for causal cross-sectional models.

This file is deliberately separate from a competition submission.  It uses
the organizer's walk-forward splits and evaluator, and every fit sees only
rows preceding its validation slice.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from data_loader import list_periods, load_period
from evaluation import backtest


def cross_sectional_rank(frame: pd.DataFrame) -> np.ndarray:
    n = frame.shape[1]
    ranked = frame.rank(axis=1, method="average").to_numpy(dtype=float)
    return (ranked - (n + 1.0) / 2.0) / max((n - 1.0) / 2.0, 1.0)


def demean(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=1, keepdims=True)


def design(features: pd.DataFrame) -> tuple[np.ndarray, list[str], list[str]]:
    tickers = [f"ticker.{i}" for i in range(1, 21)]
    rank = [
        cross_sectional_rank(features[f"Feature.{number}"][tickers])
        for number in range(1, 7)
    ]
    core = demean(sum(rank[:4]) / 4.0)
    residuals = [demean(value - core) for value in rank[:4]]
    dispersion = demean(np.mean(np.square(np.stack(residuals, axis=2)), axis=2))
    rank5, rank6 = rank[4], rank[5]

    # A compact, symmetric basis: the related first four features contribute
    # through a common component and disagreement; features 5 and 6 can have
    # independent nonlinear and interaction effects.  Every basis column is
    # cross-sectionally de-meaned before fitting.
    blocks = [
        core,
        *residuals,
        rank5,
        rank6,
        demean(core * core),
        demean(rank5 * rank5),
        demean(rank6 * rank6),
        demean(core * rank5),
        demean(core * rank6),
        demean(rank5 * rank6),
        dispersion,
    ]
    array = np.stack(blocks, axis=2)
    names = [
        "core", "resid_1", "resid_2", "resid_3", "resid_4", "rank_5", "rank_6",
        "core_sq", "rank_5_sq", "rank_6_sq", "core_x_5", "core_x_6", "rank_5_x_6",
        "core_disagreement",
    ]
    return array, tickers, names


def fit_ridge(
    feature_array: np.ndarray,
    target: np.ndarray,
    lookback: int,
    ridge: float,
) -> np.ndarray:
    if lookback > 0:
        feature_array = feature_array[-lookback:]
        target = target[-lookback:]
    valid_rows = np.std(target, axis=1) > 1e-12
    x = feature_array[valid_rows].reshape(-1, feature_array.shape[2])
    y = demean(target[valid_rows]).reshape(-1)
    scale = np.sqrt(np.mean(x * x, axis=0))
    scale[scale < 1e-8] = 1.0
    xs = x / scale
    gram = xs.T @ xs / len(xs)
    rhs = xs.T @ y / len(xs)
    beta = np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), rhs)
    return beta / scale


def smooth_positions(
    scores: np.ndarray,
    span: int,
    initial: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if span <= 1:
        return demean(scores), scores[-1] - scores[-1].mean()
    alpha = 2.0 / (span + 1.0)
    state = np.zeros(scores.shape[1], dtype=float) if initial is None else initial.copy()
    output = np.empty_like(scores)
    for row in range(len(scores)):
        state = alpha * scores[row] + (1.0 - alpha) * state
        output[row] = state
    return demean(output), state - state.mean()


def evaluate_config(
    periods: list[str],
    lookback: int,
    ridge: float,
    span: int,
    selected: set[str] | None = None,
) -> dict:
    pnl_parts: list[pd.Series] = []
    ic_values: list[float] = []
    fold_scores: list[tuple[str, float]] = []
    beta_values: list[np.ndarray] = []
    turn_values: list[float] = []

    for period in periods:
        data = load_period(period)
        n_val = min(800, int(0.2 * len(data["features"])))
        n_train = len(data["features"]) - n_val
        full_x, tickers, names = design(data["features"])
        if selected is not None:
            positions = [i for i, name in enumerate(names) if name in selected]
            full_x = full_x[:, :, positions]
        target = data["target"][tickers].to_numpy(dtype=float)
        beta = fit_ridge(full_x[:n_train], target[:n_train], lookback, ridge)
        raw_train = np.einsum("tnk,k->tn", full_x[:n_train], beta)
        _, state = smooth_positions(raw_train, span)
        raw_val = np.einsum("tnk,k->tn", full_x[n_train:], beta)
        pred_values, _ = smooth_positions(raw_val, span, state)
        predictions = pd.DataFrame(
            pred_values,
            index=data["features"].index[n_train:],
            columns=tickers,
        )
        returns = data["returns"].iloc[n_train:][tickers]
        pnl = backtest(predictions, returns)
        pnl_parts.append(pnl)
        fold_sharpe = float(pnl.mean() / pnl.std()) if pnl.std() > 0 else 0.0
        fold_scores.append((period, fold_sharpe))
        corr = predictions.corrwith(data["target"].iloc[n_train:][tickers], axis=1)
        ic_values.append(float(corr.mean()))
        turnover = predictions.shift(1).fillna(0.0).diff().abs().sum(axis=1).mean()
        turn_values.append(float(turnover))
        beta_values.append(beta)
        print(
            f"{period}: sharpe={fold_sharpe:+.4f}, ic={corr.mean():+.4f}, "
            f"turnover={turnover:.4f}"
        )

    pnl_all = pd.concat(pnl_parts, ignore_index=True)
    return {
        "sharpe": float(pnl_all.mean() / pnl_all.std()),
        "ic": float(np.mean(ic_values)),
        "turnover": float(np.mean(turn_values)),
        "fold_scores": fold_scores,
        "beta": np.mean(beta_values, axis=0),
        "fold_std": float(np.std([value for _, value in fold_scores])),
        "positive_folds": int(sum(value > 0 for _, value in fold_scores)),
        "n_folds": len(fold_scores),
    }


def evaluate_bases(periods: list[str], span: int, selected: set[str] | None = None) -> None:
    pnl_parts: dict[str, list[pd.Series]] = {}
    ic_parts: dict[str, list[float]] = {}
    fold_parts: dict[str, list[float]] = {}
    for period in periods:
        data = load_period(period)
        n_val = min(800, int(0.2 * len(data["features"])))
        n_train = len(data["features"]) - n_val
        full_x, tickers, names = design(data["features"])
        target = data["target"].iloc[n_train:][tickers]
        returns = data["returns"].iloc[n_train:][tickers]
        for column, name in enumerate(names):
            if selected is not None and name not in selected:
                continue
            for sign, suffix in ((1.0, "+"), (-1.0, "-")):
                key = f"{suffix}{name}"
                _, state = smooth_positions(sign * full_x[:n_train, :, column], span)
                values, _ = smooth_positions(sign * full_x[n_train:, :, column], span, state)
                pred = pd.DataFrame(values, index=returns.index, columns=tickers)
                pnl = backtest(pred, returns)
                pnl_parts.setdefault(key, []).append(pnl)
                ic_parts.setdefault(key, []).append(float(pred.corrwith(target, axis=1).mean()))
                fold_parts.setdefault(key, []).append(float(pnl.mean() / pnl.std()))
        print(f"{period}: completed")
    rows = []
    for key, parts in pnl_parts.items():
        pnl = pd.concat(parts, ignore_index=True)
        rows.append((float(pnl.mean() / pnl.std()), float(np.mean(ic_parts[key])), key))
    for sharpe, ic, key in sorted(rows, reverse=True):
        hits = sum(value > 0.0 for value in fold_parts[key])
        print(f"{key:24s} sharpe={sharpe:+.4f} ic={ic:+.4f} folds={hits}/{len(fold_parts[key])}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--periods", type=int, default=12)
    parser.add_argument("--lookback", type=int, default=10000)
    parser.add_argument("--ridge", type=float, default=0.05)
    parser.add_argument("--span", type=int, default=20)
    parser.add_argument("--bases", action="store_true")
    parser.add_argument("--select", type=str, default="")
    args = parser.parse_args()
    all_periods = list_periods()
    periods = all_periods[-args.periods :]
    selected = {value.strip() for value in args.select.split(",") if value.strip()}
    if args.bases:
        evaluate_bases(periods, args.span, selected or None)
        return
    result = evaluate_config(
        periods, args.lookback, args.ridge, args.span, selected or None,
    )
    print("=" * 68)
    print(
        f"overall sharpe={result['sharpe']:+.4f} ic={result['ic']:+.4f} "
        f"turnover={result['turnover']:.4f} folds={result['positive_folds']}/{result['n_folds']} "
        f"fold_std={result['fold_std']:.4f}"
    )
    print("mean beta", np.array2string(result["beta"], precision=5))


if __name__ == "__main__":
    main()
