import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
import numpy as np
import pandas as pd
from predictor import Predictor
from walkforward import walkforward_full

class BoundedRidgePredictor(Predictor):
    def __init__(self):
        self.beta = None
        self.span = 720
        self.reg = 5000.0

    def _tickers(self, features):
        return sorted(features["Feature.1"].columns, key=lambda name: int(name.split(".")[-1]))

    def _ranks(self, features, tickers):
        result = []
        for number in range(1, 7):
            values = np.nan_to_num(
                features[f"Feature.{number}"][tickers].to_numpy(float),
                nan=0.0, posinf=0.0, neginf=0.0)
            order = np.argsort(np.argsort(values, axis=1, kind="mergesort"),
                               axis=1, kind="mergesort")
            result.append((order.astype(float) - 9.5) / 9.5)
        return result

    def _design(self, features):
        tickers = self._tickers(features)
        ranks = self._ranks(features, tickers)
        core = sum(ranks[:4]) / 4.0
        primary = -core**3 + core * ranks[5]
        
        # We use primary and F1
        X = np.column_stack([
            primary.reshape(-1),
            ranks[0].reshape(-1) # F1
        ])
        return X, tickers, features.index

    def train(self, features, target):
        X, tickers, _ = self._design(features)
        y = target[tickers].values.reshape(-1)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        
        X_ok = X[ok]
        y_ok = y[ok]
        
        gram = X_ok.T @ X_ok
        rhs = X_ok.T @ y_ok
        
        reg_matrix = self.reg * np.eye(2)
        beta = np.linalg.solve(gram + reg_matrix, rhs)
        
        # THE MAGIC: Force primary weight to be positive and dominant!
        beta[0] = np.abs(beta[0]) + 0.1 # Ensure it's strictly positive and has baseline weight
        self.beta = beta

    def predict(self, features):
        X, tickers, index = self._design(features)
        
        raw = X @ self.beta
        
        pred_matrix = pd.DataFrame(
            raw.reshape(-1, len(tickers)), index=index, columns=tickers,
        )
        
        # Causal position smoothing
        pred_matrix = pred_matrix.ewm(span=self.span, min_periods=1).mean()
        
        # Causal removal of the static per-instrument tilt
        expanding_mean = pred_matrix.shift(1).expanding(min_periods=1).mean()
        pred_matrix = (pred_matrix - expanding_mean).fillna(0.0)
        
        # Final cross-sectional de-mean
        pred_matrix = pred_matrix.sub(pred_matrix.mean(axis=1), axis=0)
        return pred_matrix.fillna(0.0)

if __name__ == "__main__":
    print("Testing BoundedRidgePredictor (span=720)")
    try:
        res = walkforward_full(BoundedRidgePredictor, verbose=False)
        print(f"Sharpe (full): {res['sharpe']:.6f}")
    except Exception as e:
        print(f"Error: {e}")
