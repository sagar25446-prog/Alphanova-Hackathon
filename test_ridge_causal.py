import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from predictor import Predictor

class RidgeF5Causal(Predictor):
    def __init__(self):
        self.beta = None
        self.span = 120

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
        sleeve = -ranks[4]
        
        X = np.column_stack([
            primary.reshape(-1),
            sleeve.reshape(-1)
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
        
        reg = 1000.0 * np.eye(2)
        self.beta = np.linalg.solve(gram + reg, rhs)

    def predict(self, features):
        X, tickers, index = self._design(features)
        raw = X @ self.beta
        
        pred_matrix = pd.DataFrame(
            raw.reshape(-1, len(tickers)), index=index, columns=tickers,
        )
        
        pred_matrix = pred_matrix.ewm(span=self.span, min_periods=1).mean()
        
        expanding_mean = pred_matrix.shift(1).expanding(min_periods=1).mean()
        pred_matrix = (pred_matrix - expanding_mean).fillna(0.0)
        
        pred_matrix = pred_matrix.sub(pred_matrix.mean(axis=1), axis=0)
        return pred_matrix.fillna(0.0)

if __name__ == "__main__":
    from walkforward import walkforward_full
    print(walkforward_full(RidgeF5Causal, "data/train", "small", verbose=False))
