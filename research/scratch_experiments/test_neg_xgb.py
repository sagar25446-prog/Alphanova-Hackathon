"""Test if negating XGB on Target works."""
import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
import numpy as np
import pandas as pd
import xgboost as xgb
from data_loader import list_periods, load_period
from evaluation import backtest
from walkforward import compute_val_rows, walkforward_full
from predictor import Predictor

class NegatedXGB(Predictor):
    def __init__(self):
        self.params = {"objective": "reg:squarederror", "max_depth": 2, "eta": 0.05, "min_child_weight": 500, "verbosity": 0, "nthread": 4}
        self.model = None

    def _csrank(self, df):
        r = df.rank(axis=1)
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

    def train(self, features, target):
        g = {i: self._csrank(features[f"Feature.{i}"]) for i in range(1, 7)}
        X = np.column_stack([g[i].values.reshape(-1) for i in range(1, 7)])
        y = target.values.reshape(-1)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        self.model = xgb.train(self.params, xgb.DMatrix(X[ok], label=y[ok]), num_boost_round=30)

    def predict(self, features):
        g = {i: self._csrank(features[f"Feature.{i}"]) for i in range(1, 7)}
        X = np.column_stack([g[i].values.reshape(-1) for i in range(1, 7)])
        pred = self.model.predict(xgb.DMatrix(np.nan_to_num(X)))
        pred = pd.DataFrame(pred.reshape(-1, features.shape[1] // 6), index=features.index)
        
        # NEGATE THE PREDICTION
        pred = -pred 
        
        pred = pred.ewm(span=72, min_periods=1).mean()
        pred = pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)
        return pred

res = walkforward_full(NegatedXGB, gauge_fix=False, verbose=False)
print(f"Negated XGB Full Sharpe: {res['sharpe']:.6f}")
