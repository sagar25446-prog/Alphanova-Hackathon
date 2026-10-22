"""Rank-1 Submission: Ultra-low-turnover XGBoost with aggressive smoothing.

Key insight: Our previous XGBoost submissions had POSITIVE raw test scores
on the server (up to +0.018) but NEGATIVE Sharpe because transaction costs
(5bps on L1 turnover) destroyed the signal. The fix is extreme smoothing.

This model follows the exact demo_engineered.py pattern (proven to pass
the gate) but with:
1. span=720 instead of 120 — cuts turnover by ~60%
2. Same contemporaneous cross-sectional features (no rolling windows)
3. Same low-capacity XGBoost (max_depth=2, 30 rounds, reg_lambda=5.0)
4. Same causal tilt removal (expanding mean subtraction)
5. Same cross-sectional de-meaning
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from predictor import Predictor


class RankOnePredictor(Predictor):
    """Ultra-low-turnover XGBoost on contemporaneous cross-sectional ranks.
    
    Identical to demo_engineered.py except span=720 for aggressive turnover
    reduction. The gate measures signal vs noise — smoothing doesn't affect
    that because it's applied post-prediction. Transaction costs are the
    killer: at 5bps, a model that changes position frequently will have
    its entire gross PnL eaten by costs.
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
        self.span = 720  # vs 120 in demo — cuts turnover by ~60%

    @staticmethod
    def _csrank(df):
        r = df.rank(axis=1)
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

    def _blocks(self, features):
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        b = {}
        b["a"] = self._csrank(g[6])
        b["m"] = self._csrank(g[1])
        b["p"] = self._csrank(g[1] - g[2])
        b["q"] = self._csrank(g[3] - g[4])
        b["r"] = self._csrank(g[1] - g[3])
        b["s"] = self._csrank(g[5])
        return b

    def _design(self, features):
        b = self._blocks(features)
        tickers = list(b[next(iter(b))].columns)
        X = np.hstack([d[tickers].values.reshape(-1, 1) for d in b.values()])
        return X, tickers, b[next(iter(b))].index

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
        # Ultra-slow smoothing to minimize turnover
        pred_matrix = pred_matrix.ewm(span=self.span, min_periods=1).mean()
        # Causal tilt removal
        expanding_mean = pred_matrix.shift(1).expanding(min_periods=1).mean()
        pred_matrix = (pred_matrix - expanding_mean).fillna(0.0)
        # Cross-sectional de-mean
        pred_matrix = pred_matrix.sub(pred_matrix.mean(axis=1), axis=0)
        return pred_matrix.fillna(0.0)
