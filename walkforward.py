"""Walk-forward backtesting engine for the Bi-weekly competition, Season 1.

Two modes:

  walkforward_backtest  — Quick validation on the latest period.
                          Holds out min(800, 20%) rows for validation.
                          Returns Sharpe and equity curve.

  walkforward_full      — Full walk-forward across ALL periods. For each period,
                          holds out min(800, 20%) rows for validation.
                          Concatenates PnL across all validation sets.
                          Optionally gauge-fixes predictions to compute city, IC,
                          and other geometric metrics (requires shipped gauge
                          matrices in data/train/gauge/).

                          NOTE: These are validation-set statistics computed on
                          training data. Performance on the held-out test set
                          (which we run internally after submission) will differ.

Each period file already contains all history up to that point — there is
no need to concatenate multiple period files.

Contestants should NOT modify this file.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import list_periods, load_period
from evaluation import (
    backtest,
    check_cross_sectional_demean,
    returns_to_equity,
    sharpe_ratio,
)
from predictor import Predictor

MAX_VAL_ROWS = 800
VAL_FRACTION = 0.2


def compute_val_rows(n: int) -> int:
    """Validation holdout size: min(MAX_VAL_ROWS, 20% of n)."""
    return min(MAX_VAL_ROWS, int(VAL_FRACTION * n))


def walkforward_backtest(
    predictor_class: type,
    data_dir: str = "data/train",
    universe: str = "small",
    max_train_time: float = 240.0,
    max_predict_time: float = 60.0,
    verbose: bool = True,
) -> dict:
    """Run walk-forward backtest for a Predictor subclass.

    Uses the latest available period. Holds out min(800, 20%) rows for
    validation. The predictor trains on the rest and predicts on the
    validation portion.

    Each period file already contains all history up to that period's test start,
    so no concatenation of periods is needed.

    Args:
        predictor_class: A class that inherits from Predictor.
        data_dir: Directory containing parquet files.
        universe: Universe type ('small', 'top20vol', 'top50').
        max_train_time: Maximum training time in seconds (default 4 min).
        max_predict_time: Maximum prediction time in seconds (default 60s).
        verbose: Whether to print progress.

    Returns:
        Dictionary with:
            - 'sharpe': Sharpe ratio on the validation portion
            - 'train_rows': Number of training rows
            - 'val_rows': Number of validation rows
            - 'predictions': Validation predictions DataFrame
            - 'returns': Validation returns DataFrame
            - 'pf_returns': Portfolio returns on validation set
            - 'equity': Cumulative equity curve
            - 'demean_check': Whether de-mean check passed
    """
    if not (
        isinstance(predictor_class, type) and issubclass(predictor_class, Predictor)
    ):
        raise TypeError(
            f"{predictor_class} must be a class that inherits from Predictor"
        )

    # Use the latest period (most data)
    periods = list_periods(data_dir, universe)
    if not periods:
        raise FileNotFoundError(
            f"No training data found in '{data_dir}' for universe '{universe}'. "
            f"Check that the data directory exists and contains files matching "
            f"*.{universe}.features.parquet"
        )
    latest_period = periods[-1]

    if verbose:
        print(f"Available periods: {len(periods)} ({periods[0]} to {periods[-1]})")
        print(f"Using latest period: {latest_period}")

    data = load_period(latest_period, data_dir, universe)
    n_total = len(data["features"])
    n_val = compute_val_rows(n_total)
    n_train = n_total - n_val

    if verbose:
        print(f"Total rows: {n_total}")
        print(f"Training rows: {n_train}")
        print(f"Validation rows: {n_val} (min({MAX_VAL_ROWS}, {VAL_FRACTION:.0%}))")
        print("=" * 60)

    # Split
    train_features = data["features"].iloc[:n_train]
    train_target = data["target"].iloc[:n_train]
    train_returns = data["returns"].iloc[:n_train]

    val_features = data["features"].iloc[n_train:]
    val_target = data["target"].iloc[n_train:]
    val_returns = data["returns"].iloc[n_train:]

    # Train
    predictor = predictor_class()
    # Parity with the internal runner (review of PR #34): predictors that
    # implement generate_features() get it called on the FULL feature frame
    # before training, exactly as the scoring pipeline does after upload.
    # Without this, such submissions validate differently locally than they
    # score remotely — the gap this local kit exists to prevent.
    if hasattr(predictor, "generate_features"):
        predictor.generate_features(data["features"])

    t0 = time.time()
    try:
        predictor.train(train_features, train_target)
    except Exception as e:
        print(f"Training FAILED: {e}")
        return {"sharpe": 0.0, "error": f"train: {e}", "error_phase": "train"}
    train_time = time.time() - t0

    if train_time > max_train_time:
        print(f"WARNING: Training took {train_time:.1f}s (limit: {max_train_time}s)")
    elif verbose:
        print(f"Trained in {train_time:.1f}s")

    # Predict on validation features
    t0 = time.time()
    try:
        predictions = predictor.predict(val_features)
    except Exception as e:
        print(f"Prediction FAILED: {e}")
        return {"sharpe": 0.0, "error": f"predict: {e}", "error_phase": "predict"}
    predict_time = time.time() - t0

    if predict_time > max_predict_time:
        print(f"WARNING: Prediction took {predict_time:.1f}s (limit: {max_predict_time}s)")
    elif verbose:
        print(f"Predicted in {predict_time:.1f}s")

    dm_ok = check_cross_sectional_demean(predictions)

    # Compute backtest once — `sharpe_ratio` would call backtest again
    # internally, double-charging the alignment / ffill work for no
    # reason. Same shape as the equivalent inline calc at line ~322.
    pf_ret = backtest(predictions, val_returns)
    val_sharpe = float(pf_ret.mean() / pf_ret.std()) if pf_ret.std() > 0 else 0.0
    equity = returns_to_equity(pf_ret)

    if verbose:
        print(f"\nValidation Sharpe: {val_sharpe:.4f}")
        print(f"Equity: start={equity.iloc[0]:.4f}, end={equity.iloc[-1]:.4f}")
        print(f"Max drawdown from peak: {(equity - equity.cummax()).min():.4f}")

    return {
        "sharpe": val_sharpe,
        "train_rows": n_train,
        "val_rows": n_val,
        "predictions": predictions,
        "returns": val_returns,
        "pf_returns": pf_ret,
        "equity": equity,
        "demean_check": dm_ok,
    }


def walkforward_full(
    predictor_class: type,
    data_dir: str = "data/train",
    universe: str = "small",
    gauge_fix: bool = True,
    max_train_time: float = 240.0,
    max_predict_time: float = 60.0,
    verbose: bool = True,
) -> dict:
    """Full walk-forward across all periods with optional gauge-fixing.

    For each period:
        1. Hold out min(800, 20%) rows for validation
        2. Train a fresh predictor on the rest
        3. Predict on the validation portion
        4. Compute PnL on the validation portion
        5. If gauge_fix=True, apply shipped gauge matrices to get gauge-fixed vectors

    After all periods:
        - Concatenate PnL → compute Sharpe
        - If gauge_fix: concatenate gauge-fixed vectors → compute IC, city,
          concentration, compression_loss, and compare against published cities

    NOTE: These are validation-set statistics computed on training data.
    Performance on the held-out test set will differ.

    Args:
        predictor_class: A class that inherits from Predictor.
        data_dir: Directory containing parquet files.
        universe: Universe type ('small', 'top20vol', 'top50').
        gauge_fix: If True, compute gauge-fixed vectors and geometric metrics.
            Requires gauge matrices in {data_dir}/gauge/.
        max_train_time: Maximum training time in seconds (default 4 min).
        max_predict_time: Maximum prediction time in seconds (default 60s).
        verbose: Whether to print progress.

    Returns:
        Dictionary with:
            - 'sharpe': Sharpe ratio across all validation sets
            - 'n_periods': Number of periods processed
            - 'n_observations': Total validation rows across all periods
            - 'pf_returns': Concatenated portfolio returns
            - 'equity': Cumulative equity curve
            - 'per_period': Dict of per-period results (sharpe, n_val)

        If gauge_fix=True, also includes:
            - 'ic': Mean IC (cross-sectional correlation with target)
            - 'ic_std': IC standard deviation
            - 'ic_dispersion_corr': Correlation between IC and target dispersion
            - 'concentration': Trajectory tightness on S^18
            - 'compression_loss': IC - city[0]
            - 'city': Unit vector in R^19 (the city)
            - 'gauge_fixed': Concatenated gauge-fixed vectors (n_obs, 19)
            - 'city_comparison': DataFrame comparing against published cities
            - 'city_novelty_deg': Angular distance to nearest city (degrees)
            - 'nearest_city': ID of nearest city in the database
            - 'global_novelty_deg': NaN (computed internally after submission)
            - 'nearest_signal': N/A (computed internally after submission)
    """
    if not (
        isinstance(predictor_class, type) and issubclass(predictor_class, Predictor)
    ):
        raise TypeError(
            f"{predictor_class} must be a class that inherits from Predictor"
        )

    periods = list_periods(data_dir, universe)

    if verbose:
        print(f"Walk-forward full: {len(periods)} periods, gauge_fix={gauge_fix}")
        print(f"Validation: min({MAX_VAL_ROWS}, {VAL_FRACTION:.0%}) per period")
        print("NOTE: These are validation-set statistics. Test-set results will differ.")
        print("=" * 60)

    # Preload gauge matrices if needed
    gauge_dir = Path(data_dir) / "gauge"
    if gauge_fix:
        from city_tools import compare_to_cities, compute_city, compute_ic

        if not gauge_dir.exists():
            raise FileNotFoundError(
                f"Gauge directory {gauge_dir} not found. "
                "Gauge matrices are required for gauge_fix=True."
            )

    all_pnl = []
    all_gf = []
    all_target_dispersion = []
    per_period = {}

    for i, period in enumerate(periods):
        t_period = time.time()

        data = load_period(period, data_dir, universe)
        n_total = len(data["features"])
        n_val = compute_val_rows(n_total)
        n_train = n_total - n_val

        if n_val == 0:
            if verbose:
                print(f"  [{period}] SKIP (no validation rows)")
            continue

        train_features = data["features"].iloc[:n_train]
        train_target = data["target"].iloc[:n_train]
        val_features = data["features"].iloc[n_train:]
        val_returns = data["returns"].iloc[n_train:]
        val_target = data["target"].iloc[n_train:]

        # Train
        predictor = predictor_class()
        # Same parity hook as the single-period path above (PR #34 review).
        if hasattr(predictor, "generate_features"):
            predictor.generate_features(data["features"])
        t0 = time.time()
        try:
            predictor.train(train_features, train_target)
        except Exception as e:
            if verbose:
                print(f"  [{period}] Training FAILED: {e}")
            continue
        train_time = time.time() - t0

        if train_time > max_train_time:
            if verbose:
                print(f"  [{period}] WARNING: Training took {train_time:.1f}s")

        # Predict
        t0 = time.time()
        try:
            predictions = predictor.predict(val_features)
        except Exception as e:
            if verbose:
                print(f"  [{period}] Prediction FAILED: {e}")
            continue

        # Target dispersion: computed inside gauge block below to stay row-aligned with gf
        predict_time = time.time() - t0

        if predict_time > max_predict_time:
            if verbose:
                print(f"  [{period}] WARNING: Prediction took {predict_time:.1f}s")

        # PnL
        pf_ret = backtest(predictions, val_returns)
        period_sharpe = float(pf_ret.mean() / pf_ret.std()) if pf_ret.std() > 0 else 0.0
        all_pnl.append(pf_ret)

        # Gauge-fix
        if gauge_fix:
            gauge_path = gauge_dir / f"{period}.{universe}.gauge.npy"
            if not gauge_path.exists():
                if verbose:
                    print(f"  [{period}] WARNING: No gauge file, skipping gauge-fix")
            else:
                gauge_matrices = np.load(str(gauge_path))
                pred_vals = predictions.values
                # Ensure ticker ordering: ticker.1, ticker.2, ..., ticker.N
                if hasattr(predictions, 'columns'):
                    cols = predictions.columns.tolist()
                    numeric_order = sorted(cols, key=lambda c: int(c.split('.')[-1]))
                    if cols != numeric_order:
                        pred_vals = predictions[numeric_order].values

                # Cross-sectionally demean (should already be, but ensure)
                pred_vals = pred_vals - pred_vals.mean(axis=1, keepdims=True)

                n_gauge = gauge_matrices.shape[0]
                if n_gauge != n_val:
                    # Align to shorter
                    n_use = min(n_gauge, pred_vals.shape[0])
                    gauge_matrices = gauge_matrices[:n_use]
                    pred_vals = pred_vals[:n_use]

                from city_tools import compute_gauge_fixed
                gf = compute_gauge_fixed(pred_vals, gauge_matrices)
                all_gf.append(gf)

                # Target dispersion: cross-sectional std, trimmed to match gf rows
                dispersion = val_target.values[:gf.shape[0]].std(axis=1)
                all_target_dispersion.append(dispersion)

        per_period[period] = {
            "sharpe": period_sharpe,
            "n_val": n_val,
        }

        elapsed = time.time() - t_period
        if verbose:
            print(
                f"  [{period}] {n_val} val rows, "
                f"Sharpe={period_sharpe:+.4f}, "
                f"{elapsed:.1f}s"
            )

        del data, predictor

    # Aggregate
    if not all_pnl:
        if verbose:
            print("No periods produced results.")
        return {"sharpe": 0.0, "n_periods": 0, "n_observations": 0}

    concat_pnl = pd.concat(all_pnl, ignore_index=True)
    overall_sharpe = float(concat_pnl.mean() / concat_pnl.std()) if concat_pnl.std() > 0 else 0.0
    equity = returns_to_equity(concat_pnl)

    result = {
        "sharpe": overall_sharpe,
        "n_periods": len(per_period),
        "n_observations": len(concat_pnl),
        "pf_returns": concat_pnl,
        "equity": equity,
        "per_period": per_period,
    }

    if gauge_fix and all_gf:
        concat_gf = np.concatenate(all_gf, axis=0)
        ic_stats = compute_ic(concat_gf)
        city = compute_city(concat_gf)

        # IC-dispersion correlation
        ic_dispersion_corr = float('nan')
        if all_target_dispersion:
            concat_dispersion = np.concatenate(all_target_dispersion)
            ic_series = concat_gf[:, 0]
            n_common = min(len(ic_series), len(concat_dispersion))
            if n_common > 1:
                ic_dispersion_corr = float(
                    np.corrcoef(ic_series[:n_common], concat_dispersion[:n_common])[0, 1]
                )

        result.update({
            "ic": ic_stats["ic_mean"],
            "ic_std": ic_stats["ic_std"],
            "ic_dispersion_corr": ic_dispersion_corr,
            "concentration": ic_stats["concentration"],
            "compression_loss": ic_stats["ic_mean"] - city[0],
            "city": city,
            "gauge_fixed": concat_gf,
        })

        # Compare against published cities → city novelty
        try:
            city_comparison = compare_to_cities(city)
            result["city_comparison"] = city_comparison
            result["city_novelty_deg"] = float(city_comparison.iloc[0]["angular_distance_deg"])
            result["nearest_city"] = str(city_comparison.iloc[0]["signal_id"])
        except Exception:
            result["city_novelty_deg"] = float('nan')
            result["nearest_city"] = "N/A"

        # Global novelty is not computed here — it requires loading all signals'
        # full gauge-fixed trajectories and aligning timesteps. It is computed
        # internally after submission via compute_all_novelty().
        result["global_novelty_deg"] = float('nan')
        result["nearest_signal"] = "N/A (computed after submission)"

    if verbose:
        print("=" * 60)
        print(f"Overall Sharpe: {overall_sharpe:.4f}")
        print(f"Periods: {len(per_period)}, Observations: {len(concat_pnl)}")
        if gauge_fix and all_gf:
            print(f"IC: {result['ic']:.4f} (std={result['ic_std']:.4f})")
            print(f"IC-dispersion corr: {result['ic_dispersion_corr']:.4f}")
            print(f"Concentration: {result['concentration']:.4f}")
            print(f"Compression loss: {result['compression_loss']:.4f}")
            print(f"City novelty: {result['city_novelty_deg']:.1f} deg (nearest: {result['nearest_city']})")
            print(f"Global novelty: computed after submission")
        print()
        print("NOTE: These are validation-set statistics. Test-set results will differ.")

    return result
