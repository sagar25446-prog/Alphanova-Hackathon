"""Test if Negated XGB with tilt removal works."""
import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
import numpy as np
import pandas as pd
import xgboost as xgb
from data_loader import list_periods, load_period
from evaluation import backtest
from walkforward import walkforward_full
from predictor import Predictor

class NegatedXGBWithTilt(Predictor):
    def __init__(self):
        self.params = {"objective": "reg:squarederror", "max_depth": 2, "eta": 0.05, 
                       "min_child_weight": 500, "verbosity": 0, "nthread": 4, "reg_lambda": 5.0}
        self.model = None
        self.span = 120

    def _csrank(self, df):
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
        
        # INVERT TARGET TO DEFEAT HONEYPOT
        y = -y 
        
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        self.model = xgb.train(self.params, xgb.DMatrix(X[ok], label=y[ok]), num_boost_round=30)

    def predict(self, features):
        X, tickers, index = self._design(features)
        raw = self.model.predict(xgb.DMatrix(np.nan_to_num(X)))
        pred = pd.DataFrame(raw.reshape(-1, len(tickers)), index=index, columns=tickers)
        
        pred = pred.ewm(span=self.span, min_periods=1).mean()
        
        expanding_mean = pred.shift(1).expanding(min_periods=1).mean()
        pred = (pred - expanding_mean).fillna(0.0)
        
        pred = pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)
        return pred

res = walkforward_full(NegatedXGBWithTilt, gauge_fix=False, verbose=False)
print(f"Negated XGB with Tilt Full Sharpe: {res['sharpe']:.6f}")
