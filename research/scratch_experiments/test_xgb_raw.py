import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor
from walkforward import walkforward_full

class XGBRaw(Predictor):
    def __init__(self):
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
        self.span = 240

    @staticmethod
    def _csrank(df):
        r = df.rank(axis=1)
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

    def train(self, features, target):
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        b = {f"f{i}": self._csrank(g[i]) for i in range(1, 7)}
        
        tickers = list(b['f1'].columns)
        X = np.hstack([d[tickers].values.reshape(-1, 1) for d in b.values()])
        y = target[tickers].values.reshape(-1)
        
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        self.model = xgb.train(self.params, xgb.DMatrix(X[ok], label=y[ok]), self.n_rounds)

    def predict(self, features):
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        b = {f"f{i}": self._csrank(g[i]) for i in range(1, 7)}
        
        tickers = list(b['f1'].columns)
        index = b['f1'].index
        X = np.hstack([d[tickers].values.reshape(-1, 1) for d in b.values()])
        
        raw = self.model.predict(xgb.DMatrix(np.nan_to_num(X)))
        pred = pd.DataFrame(raw.reshape(-1, len(tickers)), index=index, columns=tickers)
        
        pred = pred.ewm(span=self.span, min_periods=1).mean()
        expanding_mean = pred.shift(1).expanding(min_periods=1).mean()
        pred = (pred - expanding_mean).fillna(0.0)
        return pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)

print("Testing XGBoost with RAW Features only (1-6):")
res = walkforward_full(XGBRaw, gauge_fix=False, verbose=False)
print(f"Overall Sharpe: {res['sharpe']:.4f}")
