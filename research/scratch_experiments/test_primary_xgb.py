import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor
from walkforward import walkforward_full

class PrimaryOnlyXGB(Predictor):
    def __init__(self):
        self.model = None
        self.params = {"objective": "reg:squarederror", "max_depth": 2, "eta": 0.05,
                       "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 500,
                       "reg_lambda": 5.0, "verbosity": 0, "nthread": 4}
        self.n_rounds = 30; self.span = 720
    def _design(self, features):
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        tickers = sorted(g[1].columns, key=lambda name: int(name.split(".")[-1]))
        ranks = []
        for i in range(1, 7):
            v = np.nan_to_num(g[i][tickers].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
            o = np.argsort(np.argsort(v, axis=1, kind="mergesort"), axis=1, kind="mergesort")
            ranks.append((o.astype(float) - 9.5) / 9.5)
        core = sum(ranks[:4]) / 4.0
        b = {"primary": pd.DataFrame(-core**3 + core * ranks[5], index=g[1].index, columns=tickers),
             "negcore3": pd.DataFrame(-core**3, index=g[1].index, columns=tickers)}
        X = np.hstack([d[tickers].values.reshape(-1, 1) for d in b.values()])
        return X, tickers, b["primary"].index
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
        em = pred.shift(1).expanding(min_periods=1).mean()
        pred = (pred - em).fillna(0.0)
        return pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)

if __name__ == "__main__":
    res = walkforward_full(PrimaryOnlyXGB, gauge_fix=False, verbose=False)
    print(f"PrimaryOnlyXGB Full Sharpe: {res['sharpe']:.6f}")
