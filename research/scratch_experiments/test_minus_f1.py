"""Test -F1 across all periods."""
import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
import numpy as np
import pandas as pd
from predictor import Predictor
from walkforward import walkforward_full

class MinusF1(Predictor):
    def __init__(self):
        self.span = 72
        
    def _csrank(self, df):
        r = df.rank(axis=1)
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

    def train(self, features, target):
        pass

    def predict(self, features):
        f1 = self._csrank(features["Feature.1"])
        pred = -f1
        pred = pred.ewm(span=self.span, min_periods=1).mean()
        pred = pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)
        return pred

res = walkforward_full(MinusF1, gauge_fix=False, verbose=False)
print(f"-F1 Full Sharpe: {res['sharpe']:.6f}")
print(f"-F1 Full IC: {res['ic']:.6f}")
