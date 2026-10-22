"""Rank-1 Submission V2: Expanded features + ultra-low turnover.

Same philosophy as V1 but with expanded feature set:
- All 6 individual features as cross-sectional ranks
- All 15 pairwise differences as cross-sectional ranks
- Total: 21 features instead of 6
- span=720 for ultra-low turnover
- Even heavier regularization (reg_lambda=10.0, min_child_weight=1000)
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from predictor import Predictor


class RankOnePredictorV2(Predictor):
    """Expanded-feature ultra-low-turnover XGBoost."""

    def __init__(self):
        self.model = None
        self.params = {
            "objective": "reg:squarederror",
            "max_depth": 2,
            "eta": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.5,  # Lower since more features
            "min_child_weight": 1000,  # Heavier regularization
            "reg_lambda": 10.0,        # Heavier regularization
            "verbosity": 0,
            "nthread": 4,
        }
        self.n_rounds = 30
        self.span = 720

    @staticmethod
    def _csrank(df):
        r = df.rank(axis=1)
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

    def _blocks(self, features):
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        b = {}
        # All 6 individual features
        for i in range(1, 7):
            b[f"f{i}"] = self._csrank(g[i])
        # All 15 pairwise differences
        for i in range(1, 7):
            for j in range(i + 1, 7):
                b[f"d{i}{j}"] = self._csrank(g[i] - g[j])
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
        pred_matrix = pred_matrix.ewm(span=self.span, min_periods=1).mean()
        expanding_mean = pred_matrix.shift(1).expanding(min_periods=1).mean()
        pred_matrix = (pred_matrix - expanding_mean).fillna(0.0)
        pred_matrix = pred_matrix.sub(pred_matrix.mean(axis=1), axis=0)
        return pred_matrix.fillna(0.0)
