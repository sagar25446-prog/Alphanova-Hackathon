"""Evaluation and scoring utilities for the Bi-weekly competition, Season 1 (Walk-Forward, biweekly Sharpe, 5bp cost).

Do not modify this file.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Transaction cost charged on per-period L1 turnover (Bi-weekly competition).
# 5 basis points = 0.0005 in return units. Your local Sharpe is computed
# net of this cost, matching the contest scorer.
TRANSACTION_COST = 0.0005


def backtest(predictions: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """Compute portfolio returns from signal predictions and actual returns.

    At each timestamp, the portfolio return is the dot product of the
    previous period's signal with the current period's returns:
        r(i) = <P(i-1), X(i)>

    Bi-weekly competition returns semantics: the shipped `returns.parquet` panel is
    NOT raw price returns — it is the competition's own return series, and
    backtesting signal x returns reproduces the contest's gross PnL
    directly. Early rows of a month's panel can be exactly 0.0 (the series
    construction needs a warm-up window of history before its first value);
    a flat stretch at the start of the panel is expected, not a data bug.

    Args:
        predictions: Cross-sectionally de-meaned signal (n_timestamps, n_tickers).
        returns: One-period (hourly) contest returns (n_timestamps, n_tickers).

    Returns:
        Portfolio returns series.
    """
    predictions = predictions.copy()
    predictions.ffill(inplace=True)

    # Align predictions to returns by row position. Predictors aren't
    # required to preserve the input index — `predict()` may legitimately
    # return a default-indexed (0..N-1) DataFrame even when `returns` has
    # a DatetimeIndex, or returns may have been sliced into a non-zero
    # offset (e.g. validation rows n_train..N keep their original row
    # labels). Without this, pandas' label join produces all-NaN, then
    # `.sum(skipna=True)` silently zeros the PnL — a valid predictor
    # would score Sharpe = 0. Mirrors the fix in the internal
    # evaluation.backtest so scientists testing locally see the same
    # number the internal scorer produces.
    if not predictions.index.equals(returns.index):
        if len(predictions) != len(returns):
            raise ValueError(
                f"predictions has {len(predictions)} rows but returns has "
                f"{len(returns)} rows; backtest requires the same row count "
                f"(columns must match by label)"
            )
        predictions.index = returns.index

    pf_returns = (predictions.shift(1) * returns).sum(axis=1)

    # Transaction cost (Bi-weekly competition): 5 bps on the L1 turnover between
    # consecutive held positions (|P(i-1) - P(i-2)|). Flat before the
    # first signal, so the initial entry is charged. Returns stay 1-hour.
    turnover = predictions.shift(1).fillna(0.0).diff().abs().sum(axis=1)
    pf_returns = pf_returns - TRANSACTION_COST * turnover

    pf_returns.iloc[0] = 0

    return pf_returns


def sharpe_ratio(predictions: pd.DataFrame, returns: pd.DataFrame) -> float:
    """Compute the Sharpe ratio (utility function U).

    U = E[r(i)] / sqrt(Var[r(i)])

    Args:
        predictions: Cross-sectionally de-meaned signal.
        returns: One-period returns.

    Returns:
        Sharpe ratio.
    """
    pf_returns = backtest(predictions, returns)
    if pf_returns.std() == 0:
        return 0.0
    return pf_returns.mean() / pf_returns.std()


def returns_to_equity(returns: pd.Series) -> pd.Series:
    """Convert returns series to cumulative equity curve.

    Args:
        returns: Portfolio returns series.

    Returns:
        Cumulative sum (equity curve).
    """
    return returns.cumsum()


def check_cross_sectional_demean(predictions: pd.DataFrame) -> bool:
    """Verify that predictions are cross-sectionally de-meaned.

    Requirement at each timestamp:
        - Mean across tickers ~= 0 (i.e. sum across tickers ~= 0)

    Args:
        predictions: Signal DataFrame (n_timestamps, n_tickers).

    Returns:
        True if the de-mean check passes.
    """
    row_means = predictions.mean(axis=1)
    worst_row_mean = row_means.abs().max()
    mean_ok = worst_row_mean < 1e-6

    if mean_ok:
        print("Cross-sectional de-mean check: PASS")
    else:
        print("Cross-sectional de-mean check: FAIL")
        print(f"  Worst row mean: {worst_row_mean:.8f} (should be ~0)")

    return mean_ok


def plot_walkforward_results(
    period_pf_returns: dict[str, pd.Series],
    title: str = "Walk-Forward Backtest",
) -> None:
    """Plot walk-forward equity curve from per-period portfolio returns.

    Args:
        period_pf_returns: Dict mapping period key to portfolio returns series.
        title: Plot title.
    """
    all_returns = pd.concat(period_pf_returns.values(), ignore_index=True)
    equity = returns_to_equity(all_returns)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    equity.plot(ax=axes[0], title=f"{title} — Equity Curve")
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Cumulative Return")

    all_returns.plot(ax=axes[1], title=f"{title} — Period Returns", alpha=0.5)
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("Return")

    plt.tight_layout()
    plt.show()


def calculate_metrics(
    predictions: pd.DataFrame, target: pd.DataFrame
) -> dict[str, float]:
    """Calculate performance metrics.

    Args:
        predictions: Predicted values.
        target: True target values.

    Returns:
        Dictionary with mse, mae, rmse, correlation.
    """
    pred_vals = predictions.values.flatten()
    target_vals = target.values.flatten()

    mask = ~(np.isnan(pred_vals) | np.isnan(target_vals))
    pred_vals = pred_vals[mask]
    target_vals = target_vals[mask]

    mse = np.mean((pred_vals - target_vals) ** 2)
    mae = np.mean(np.abs(pred_vals - target_vals))
    rmse = np.sqrt(mse)
    correlation = (
        np.corrcoef(pred_vals, target_vals)[0, 1] if len(pred_vals) > 1 else 0.0
    )

    return {"mse": mse, "mae": mae, "rmse": rmse, "correlation": correlation}
