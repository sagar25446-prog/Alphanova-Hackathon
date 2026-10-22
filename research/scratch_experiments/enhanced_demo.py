"""Enhanced demo_engineered: adds primary formula features to the proven gate-passing template."""
import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor


class EnhancedDemoPredictor(Predictor):
    """demo_engineered + primary formula features + tuned hyperparams."""

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
        self.span = 120

    @staticmethod
    def _csrank(df):
        r = df.rank(axis=1)
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

    def _blocks(self, features):
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        b = {}
        # Original demo features
        b["a"] = self._csrank(g[6])
        b["m"] = self._csrank(g[1])
        b["p"] = self._csrank(g[1] - g[2])
        b["q"] = self._csrank(g[3] - g[4])
        b["r"] = self._csrank(g[1] - g[3])
        b["s"] = self._csrank(g[5])
        
        # Additional features from our analysis
        ranks = {i: self._csrank(g[i]) for i in range(1, 7)}
        core = sum(ranks[i] for i in range(1, 5)) / 4.0
        b["core"] = core
        b["negcore3"] = -core**3
        b["primary"] = -core**3 + core * ranks[6]
        b["f1xf6"] = ranks[1] * ranks[6]
        b["f56diff"] = self._csrank(g[5] - g[6])
        
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
        assert self.model is not None
        X, tickers, index = self._design(features)
        raw = self.model.predict(xgb.DMatrix(np.nan_to_num(X)))
        pred = pd.DataFrame(
            raw.reshape(-1, len(tickers)), index=index, columns=tickers,
        )
        # EWM smoothing
        pred = pred.ewm(span=self.span, min_periods=1).mean()
        # Causal tilt removal
        expanding_mean = pred.shift(1).expanding(min_periods=1).mean()
        pred = (pred - expanding_mean).fillna(0.0)
        # De-mean
        pred = pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)
        return pred
