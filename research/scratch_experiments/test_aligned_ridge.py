"""Ridge predictor using aligned feature differences."""
import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")

import numpy as np
import pandas as pd
from predictor import Predictor
from walkforward import walkforward_full

class AlignedRidgePredictor(Predictor):
    def __init__(self):
        self.beta = None
        self.span = 720
        self.reg = 1000.0

    @staticmethod
    def _csrank(df):
        r = df.rank(axis=1)
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

    def _design(self, features):
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        tickers = sorted(g[1].columns, key=lambda name: int(name.split(".")[-1]))

        r = {i: self._csrank(g[i]) for i in range(1, 7)}
        
        b = {}
        b["f1_f6"] = r[1] - r[6]
        b["f2_f6"] = r[2] - r[6]
        b["f3_f6"] = r[3] - r[6]
        b["f4_f6"] = r[4] - r[6]
        b["f5_f6"] = r[5] - r[6]

        X = np.hstack([b[k][tickers].values.reshape(-1, 1) for k in b.keys()])
        return X, tickers, b["f1_f6"].index

    def train(self, features, target):
        X, tickers, _ = self._design(features)
        y = target[tickers].values.reshape(-1)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        
        X_ok = X[ok]
        y_ok = y[ok]
        
        # Ridge regression
        gram = X_ok.T @ X_ok / max(len(X_ok), 1)
        rhs = X_ok.T @ y_ok / max(len(X_ok), 1)
        self.beta = np.linalg.solve(gram + self.reg * np.eye(X_ok.shape[1]), rhs)

    def predict(self, features):
        X, tickers, index = self._design(features)
        X = np.nan_to_num(X)
        raw = X @ self.beta
        
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
    res = walkforward_full(AlignedRidgePredictor, gauge_fix=False, verbose=False)
    print(f"Aligned Ridge Full Sharpe: {res['sharpe']:.6f}")
    
    pp = res.get('per_period', {})
    recent_keys = sorted(pp.keys())[-9:]
    recent_avg = np.mean([pp[k]['sharpe'] for k in recent_keys])
    print(f"Aligned Ridge Recent 9 Sharpe: {recent_avg:.6f}")
