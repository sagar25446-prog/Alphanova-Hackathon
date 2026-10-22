import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
import numpy as np
import pandas as pd
from predictor import Predictor
from data_loader import load_period
from evaluation import backtest
from walkforward import compute_val_rows
import xgboost as xgb

class DemoNoTilt(Predictor):
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
        assert self.model is not None
        X, tickers, index = self._design(features)
        raw = self.model.predict(xgb.DMatrix(np.nan_to_num(X)))
        pred_matrix = pd.DataFrame(
            raw.reshape(-1, len(tickers)), index=index, columns=tickers,
        )
        pred_matrix = pred_matrix.ewm(span=self.span, min_periods=1).mean()
        # NO CAUSAL TILT REMOVAL!
        pred_matrix = pred_matrix.sub(pred_matrix.mean(axis=1), axis=0)
        return pred_matrix.fillna(0.0)

from walkforward import walkforward_full
print("Testing Demo WITHOUT Causal Tilt Removal:")
res = walkforward_full(DemoNoTilt, gauge_fix=False, verbose=False)
print(f"Overall Sharpe: {res['sharpe']:.4f}")

# Test with inverted target!
class DemoNoTiltInverted(DemoNoTilt):
    def train(self, features, target):
        X, tickers, _ = self._design(features)
        y = target[tickers].values.reshape(-1)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        self.model = xgb.train(
            self.params, xgb.DMatrix(X[ok], label=-y[ok]),  # INVERTED!
            num_boost_round=self.n_rounds,
        )

print("\nTesting Demo WITHOUT Causal Tilt Removal AND Inverted Target:")
res2 = walkforward_full(DemoNoTiltInverted, gauge_fix=False, verbose=False)
print(f"Overall Sharpe: {res2['sharpe']:.4f}")

