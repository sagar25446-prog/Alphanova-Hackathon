"""Test model using only aligned features."""
import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")

import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor
from walkforward import walkforward_full

class AlignedPredictor(Predictor):
    def __init__(self):
        self.model = None
        self.params = {
            "objective": "reg:squarederror",
            "max_depth": 2,
            "eta": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 1.0,
            "min_child_weight": 500,
            "reg_lambda": 5.0,
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

    def _design(self, features):
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        tickers = sorted(g[1].columns, key=lambda name: int(name.split(".")[-1]))

        # Aligned features only:
        # F1-F6, F3-F6, F4-F6, F5-F6
        r = {i: self._csrank(g[i]) for i in range(1, 7)}
        
        b = {}
        b["f1_f6"] = self._csrank(r[1] - r[6])
        b["f3_f6"] = self._csrank(r[3] - r[6])
        b["f4_f6"] = self._csrank(r[4] - r[6])
        b["f5_f6"] = self._csrank(r[5] - r[6])

        X = np.hstack([d[tickers].values.reshape(-1, 1) for d in b.values()])
        return X, tickers, b["f1_f6"].index

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
        pred = pd.DataFrame(raw.reshape(-1, len(tickers)), index=index, columns=tickers)
        
        # Smooth
        pred = pred.ewm(span=self.span, min_periods=1).mean()
        # Causal tilt removal
        em = pred.shift(1).expanding(min_periods=1).mean()
        pred = (pred - em).fillna(0.0)
        # Demean
        pred = pred.sub(pred.mean(axis=1), axis=0)
        return pred.fillna(0.0)

if __name__ == "__main__":
    res = walkforward_full(AlignedPredictor, gauge_fix=False, verbose=False)
    print(f"Aligned Predictor Full Sharpe: {res['sharpe']:.6f}")
    
    # Print recent 9 periods
    pp = res.get('per_period', {})
    recent_keys = sorted(pp.keys())[-9:]
    recent_avg = np.mean([pp[k]['sharpe'] for k in recent_keys])
    print(f"Aligned Predictor Recent 9 Sharpe: {recent_avg:.6f}")
