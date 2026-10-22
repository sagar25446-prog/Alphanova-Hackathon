import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")

import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor
from walkforward import walkforward_backtest

class EnhancedXGBSmooth(Predictor):
    def __init__(self):
        self.model = None
        self.params = {"objective": "reg:squarederror", "max_depth": 2, "eta": 0.05,
                       "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 500,
                       "reg_lambda": 5.0, "verbosity": 0, "nthread": 4}
        self.n_rounds = 30
        self.span = 720

    @staticmethod
    def _csrank(df):
        r = df.rank(axis=1)
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

    def _blocks(self, features):
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        tickers = sorted(g[1].columns, key=lambda name: int(name.split(".")[-1]))
        
        ranks = []
        for i in range(1, 7):
            vals = np.nan_to_num(g[i][tickers].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
            order = np.argsort(np.argsort(vals, axis=1, kind="mergesort"), axis=1, kind="mergesort")
            ranks.append((order.astype(float) - 9.5) / 9.5)
        
        core = sum(ranks[:4]) / 4.0
        primary = -core**3 + core * ranks[5]
        neg_core_cubed = -core**3
        
        b = {}
        b["a"] = self._csrank(g[6])
        b["m"] = self._csrank(g[1])
        b["p"] = self._csrank(g[1] - g[2])
        b["q"] = self._csrank(g[3] - g[4])
        b["r"] = self._csrank(g[1] - g[3])
        b["s"] = self._csrank(g[5])
        b["primary"] = pd.DataFrame(primary, index=g[1].index, columns=tickers)
        b["negcore3"] = pd.DataFrame(neg_core_cubed, index=g[1].index, columns=tickers)
        
        return b, tickers

    def _design(self, features):
        b, tickers = self._blocks(features)
        X = np.hstack([d[tickers].values.reshape(-1, 1) for d in b.values()])
        return X, tickers, b[next(iter(b))].index

    def train(self, features, target):
        X, tickers, _ = self._design(features)
        y = target[tickers].values.reshape(-1)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        self.model = xgb.train(self.params, xgb.DMatrix(X[ok], label=y[ok]), num_boost_round=self.n_rounds)

    def predict(self, features):
        X, tickers, index = self._design(features)
        raw = self.model.predict(xgb.DMatrix(np.nan_to_num(X)))
        pred = pd.DataFrame(raw.reshape(-1, len(tickers)), index=index, columns=tickers)
        pred = pred.ewm(span=self.span, min_periods=1).mean()
        expanding_mean = pred.shift(1).expanding(min_periods=1).mean()
        pred = (pred - expanding_mean).fillna(0.0)
        pred = pred.sub(pred.mean(axis=1), axis=0)
        return pred.fillna(0.0)

if __name__ == "__main__":
    print("Testing single-period backtest...")
    res = walkforward_backtest(EnhancedXGBSmooth, verbose=False)
    print(f"Sharpe: {res['sharpe']:.6f}")
