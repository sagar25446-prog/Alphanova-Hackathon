"""F5-Engineered XGBoost v2: heavier smoothing for lower turnover.

Same as f5_engineered but:
- span=240 (double) for lower turnover
- reg_lambda=8 (slightly stronger regularization)
"""

# /// script
# dependencies = [
#   "xgboost>=3.2.0",
# ]
# ///

import numpy as np
import pandas as pd
import xgboost as xgb

from predictor import Predictor


class F5EngineeredV2Predictor(Predictor):
    """Gate-passing XGBoost with F5 features and heavy smoothing."""

    def __init__(self):
        self.model = None
        self.params = {
            "objective": "reg:squarederror",
            "max_depth": 2,
            "eta": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 500,
            "reg_lambda": 8.0,
            "verbosity": 0,
            "nthread": 4,
        }
        self.n_rounds = 30
        self.span = 240

    @staticmethod
    def _csrank(df):
        r = df.rank(axis=1)
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

    def _blocks(self, features):
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        b = {}
        b["f5"] = self._csrank(g[5])
        b["f6"] = self._csrank(g[6])
        b["core"] = self._csrank((g[1] + g[2] + g[3] + g[4]) / 4.0)
        b["f1"] = self._csrank(g[1])
        b["d12"] = self._csrank(g[1] - g[2])
        b["d34"] = self._csrank(g[3] - g[4])
        b["d13"] = self._csrank(g[1] - g[3])
        b["f5_core"] = self._csrank(g[5] - (g[1] + g[2] + g[3] + g[4]) / 4.0)
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
