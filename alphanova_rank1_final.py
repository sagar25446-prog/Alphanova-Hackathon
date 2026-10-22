"""AlphaNova Rank1 Submission - Enhanced XGBoost with Primary Formula Features.

Gate-passing XGBoost on contemporaneous cross-sectional features enriched with
the high-IC primary formula (-core^3 + core*rank6) and negative-core-cubed.

Key design choices (data-driven):
  1. Contemporaneous cross-sectional features ONLY - no rolling windows.
     Rolling features inflate the permuted-label null and fail the gate.
  2. Primary formula (-core^3 + core*rank6) has IC=+0.074 with target OOS,
     the strongest single contemporaneous feature discovered.
  3. Low-capacity XGBoost (max_depth=2, min_child_weight=500, reg_lambda=5.0,
     30 rounds) - identical regularization to demo_engineered.py which is
     proven to pass the gate.
  4. Aggressive EWM smoothing (span=720) - reduces hourly turnover churn
     that the 5bp transaction cost taxes. Higher spans consistently improve
     net Sharpe.
  5. Causal tilt removal via expanding mean subtraction - removes the static
     long/short bias that is the easiest thing to memorize from training data.

Run:
  python runner.py alphanova_rank1_final.py
  python runner.py alphanova_rank1_final.py --full
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
from predictor import Predictor
from walkforward import walkforward_backtest

warnings.filterwarnings("ignore")

DATA_DIR = "data/train"
UNIVERSE = "small"


class EnhancedPrimaryXGBoostPredictor(Predictor):
    """XGBoost on demo features + primary formula features.

    The primary formula (-core^3 + core*rank6) captures a nonlinear
    cross-sectional interaction between the mean of features 1-4 and
    feature 6. It has the highest OOS IC with the target (+0.074)
    of any contemporaneous feature combination tested.

    All features are contemporaneous cross-sectional ranks - no rolling
    windows, no temporal state. This is critical for passing the gate.
    """

    def __init__(self):
        self.model = None
        self.params = {
            "objective": "reg:squarederror",
            "max_depth": 2,
            "eta": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 500,
            "reg_lambda": 5.0,
            "verbosity": 0,
            "nthread": 4,
        }
        self.n_rounds = 30
        self.span = 720

    @staticmethod
    def _csrank(df):
        """Cross-sectional rank normalized to [-1, 1], NaN-aware per row."""
        r = df.rank(axis=1)
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

    def _design(self, features):
        """Build the design matrix from contemporaneous cross-sectional features."""
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        tickers = sorted(
            g[1].columns, key=lambda name: int(name.split(".")[-1])
        )

        # Numpy-based ranks for nonlinear features
        ranks = []
        for i in range(1, 7):
            vals = np.nan_to_num(
                g[i][tickers].to_numpy(float),
                nan=0.0, posinf=0.0, neginf=0.0,
            )
            order = np.argsort(
                np.argsort(vals, axis=1, kind="mergesort"),
                axis=1, kind="mergesort",
            )
            ranks.append((order.astype(float) - 9.5) / 9.5)

        core = sum(ranks[:4]) / 4.0
        primary = -core**3 + core * ranks[5]   # IC=+0.074 OOS
        negcore3 = -core**3                     # IC=+0.079 OOS

        # Feature blocks
        b = {}
        # Original demo_engineered features (proven gate-passing)
        b["f6_rank"] = self._csrank(g[6])
        b["f1_rank"] = self._csrank(g[1])
        b["f1_f2_diff"] = self._csrank(g[1] - g[2])
        b["f3_f4_diff"] = self._csrank(g[3] - g[4])
        b["f1_f3_diff"] = self._csrank(g[1] - g[3])
        b["f5_rank"] = self._csrank(g[5])
        # Enhanced features: primary formula components
        b["primary"] = pd.DataFrame(
            primary, index=g[1].index, columns=tickers,
        )
        b["negcore3"] = pd.DataFrame(
            negcore3, index=g[1].index, columns=tickers,
        )

        X = np.hstack(
            [d[tickers].values.reshape(-1, 1) for d in b.values()]
        )
        return X, tickers, b["f6_rank"].index

    def train(self, features, target):
        X, tickers, _ = self._design(features)
        y = target[tickers].values.reshape(-1)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        self.model = xgb.train(
            self.params, xgb.DMatrix(X[ok], label=y[ok]),
            num_boost_round=self.n_rounds,
        )

    def predict(self, features):
        assert self.model is not None, "Must call train() before predict()"
        X, tickers, index = self._design(features)
        raw = self.model.predict(xgb.DMatrix(np.nan_to_num(X)))
        pred_matrix = pd.DataFrame(
            raw.reshape(-1, len(tickers)), index=index, columns=tickers,
        )
        # Aggressive EWM smoothing - span=720 minimizes turnover cost
        pred_matrix = pred_matrix.ewm(span=self.span, min_periods=1).mean()
        # Causal tilt removal - subtract expanding mean of past predictions
        expanding_mean = pred_matrix.shift(1).expanding(min_periods=1).mean()
        pred_matrix = (pred_matrix - expanding_mean).fillna(0.0)
        # Final cross-sectional de-mean (required by competition)
        pred_matrix = pred_matrix.sub(pred_matrix.mean(axis=1), axis=0)
        return pred_matrix.fillna(0.0)


if __name__ == "__main__":
    periods = list_periods(DATA_DIR, UNIVERSE)
    print(f"Available training periods: {len(periods)}")

    results = walkforward_backtest(
        EnhancedPrimaryXGBoostPredictor,
        data_dir=DATA_DIR,
        universe=UNIVERSE,
        verbose=True,
    )

    print(f"\nValidation Sharpe: {results['sharpe']:.4f}")
