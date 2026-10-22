"""Bi-weekly competition Season 1: Walk-Forward Cross-Sectional Signal Forecasting

This script demonstrates how to:
  1. Load and explore the obfuscated training data
  2. Implement a Predictor subclass (XGBoost example)
  3. Run walk-forward validation (retrain each period on expanding window)
  4. Evaluate your signal using the Sharpe ratio utility

Walk-forward process:
  The training data is split into an earlier training portion and a later
  validation portion. For each validation period, a fresh predictor is
  trained on all prior data and used to predict on the held-out period.

Your submission must be a subclass of Predictor with train() and predict()
methods. All logic must live inside your class. Do not modify any .py files.
"""

# /// script
# dependencies = [
#   "xgboost>=3.2.0",
# ]
# ///

import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

from data_loader import list_periods, load_period
from evaluation import returns_to_equity
from predictor import Predictor
from walkforward import walkforward_backtest

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════
# 1. DATA OVERVIEW (see comments below)
# ══════════════════════════════════════════════════════════════════════════

DATA_DIR = "data/train"
UNIVERSE = "small"

# KEY NOTES:
# - Each period is independently obfuscated — ticker.1 in one period is NOT
#   the same asset as ticker.1 in another period
# - Target is cross-sectionally de-meaned
# - Your predictions must also be cross-sectionally de-meaned
# - Pass sample["returns"] (loaded below) to evaluation.sharpe_ratio() to
#   compute a local Sharpe matching the contest's score

# ══════════════════════════════════════════════════════════════════════════
# 2. DEFINE YOUR PREDICTOR
# ══════════════════════════════════════════════════════════════════════════
#
# Your submission must inherit from Predictor and implement:
#   - train(features, target): train your model
#   - predict(features): return de-meaned predictions
#
# All code must be inside your class. Do not define helper functions
# outside of it.


class XGBoostPredictor(Predictor):
    """Example predictor using XGBoost with cross-sectional correlation objective.

    This is a simple baseline. You are encouraged to experiment with:
    - Different models and hyperparameters
    - Feature engineering (rolling stats, cross-sectional ranks, etc.)
    - Ensemble methods
    - Different objective functions

    Requirements:
    - All code must be inside this class
    - train() receives features (MultiIndex) and target (de-meaned)
    - predict() must return cross-sectionally de-meaned predictions
    """

    def __init__(self):
        self.model = None
        self.trained = False
        self.params = {
            "max_depth": 4,
            "eta": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "verbosity": 0,
        }
        self.n_rounds = 50

    def _cross_sectional_corr_objective(self, y_pred, dtrain):
        """Custom objective: maximize cross-sectional correlation.

        Gradient is the negative de-meaned target (pushing predictions
        toward the cross-sectional ranking of the target).
        """
        y_true = dtrain.get_label()
        n_assets = self._n_assets

        y_true_2d = y_true.reshape(-1, n_assets)
        r_centered = y_true_2d - y_true_2d.mean(axis=1, keepdims=True)

        grad = -r_centered.ravel()
        hess = np.ones_like(grad)
        return grad, hess

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """Train XGBoost on stacked cross-sectional data."""
        self._n_assets = target.shape[1]

        # Stack: (time, ticker) → rows, features → columns
        X = features.stack(level=1)
        # Align target rows to X's row order before the positional pairing in
        # DMatrix below: the shipped features are column-sorted (lexicographic
        # ticker order) while target ships in ticker-map order, and stack()
        # preserves each frame's own order — without the reindex every row's
        # label belongs to a DIFFERENT ticker.
        y = target.stack().reindex(X.index)

        mask = ~(y.isna() | np.isinf(y))
        X = X.loc[mask]
        y = y.loc[mask]

        # Replace inf/nan in features
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        dtrain = xgb.DMatrix(X.values, label=y.values)

        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.n_rounds,
            obj=self._cross_sectional_corr_objective,
        )
        self.trained = True

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate cross-sectionally de-meaned predictions."""
        assert self.trained, "Must call train() before predict()"

        tickers = features.columns.get_level_values(1).unique().tolist()
        n_assets = len(tickers)
        index = features.index

        X = features.stack(level=1)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        raw_preds = self.model.predict(xgb.DMatrix(X.values))

        # Reshape back to (time, ticker)
        pred_matrix = pd.DataFrame(
            raw_preds.reshape(-1, n_assets),
            index=index,
            columns=tickers,
        )

        # Cross-sectional de-mean (required)
        pred_matrix = pred_matrix.sub(pred_matrix.mean(axis=1), axis=0)
        pred_matrix = pred_matrix.fillna(0.0)

        return pred_matrix


# ══════════════════════════════════════════════════════════════════════════
# 3. VALIDATION & SUBMISSION
# ══════════════════════════════════════════════════════════════════════════
#
# Quick validation (latest period):
#   python runner.py demo.py
#
# Full walk-forward (all periods, PnL only):
#   python runner.py demo.py --full
#
# Full walk-forward + geometric metrics (city, IC, concentration):
#   python runner.py demo.py --gauge-fix
#
# NOTE: Your final submission will be scored on hidden test data that is
# NOT included here. Validation uses training data only — test-set
# results will differ.

if __name__ == "__main__":
    # Explore data
    periods = list_periods(DATA_DIR, UNIVERSE)
    print(f"Available training periods: {len(periods)}")
    print(f"  First: {periods[0]},  Last: {periods[-1]}")

    sample = load_period(periods[0], DATA_DIR, UNIVERSE)
    print(f"\nSample period: {periods[0]}")
    print(f"  Features shape: {sample['features'].shape}")
    print(f"  Returns shape:  {sample['returns'].shape}")
    print(f"  Target shape:   {sample['target'].shape}")
    print(
        f"\n  Feature names: "
        f"{sample['features'].columns.get_level_values(0).unique().tolist()}"
    )
    print(f"  Tickers:       {sample['returns'].columns.tolist()[:5]}...")

    # Quick validation
    print("\n" + "=" * 60)
    print("VALIDATION (latest period)")
    print("=" * 60)

    results = walkforward_backtest(
        XGBoostPredictor,
        data_dir=DATA_DIR,
        universe=UNIVERSE,
        verbose=True,
    )

    print(f"\nValidation Sharpe: {results['sharpe']:.4f}")
    print(f"De-mean check: {'PASS' if results['demean_check'] else 'FAIL'}")
