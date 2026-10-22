"""Sweep XGBoost hyperparameters for maximizing out-of-sample Sharpe."""
import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")

import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor
from walkforward import walkforward_full


class TunedPredictor(Predictor):
    """XGBoost predictor with tunable parameters."""
    SPAN = 480
    DEPTH = 2
    ROUNDS = 30
    LAMBDA = 5.0
    MCW = 500
    ETA = 0.05
    FEATURES = "full"  # "demo", "full", "minimal"

    def __init__(self):
        self.model = None
        self.params = {
            "objective": "reg:squarederror",
            "max_depth": self.__class__.DEPTH,
            "eta": self.__class__.ETA,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": self.__class__.MCW,
            "reg_lambda": self.__class__.LAMBDA,
            "verbosity": 0,
            "nthread": 4,
        }
        self.n_rounds = self.__class__.ROUNDS
        self.span = self.__class__.SPAN

    @staticmethod
    def _csrank(df):
        r = df.rank(axis=1)
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

    def _blocks(self, features):
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        b = {}
        feat_type = self.__class__.FEATURES
        
        if feat_type == "demo":
            # Same as demo_engineered
            b["a"] = self._csrank(g[6])
            b["m"] = self._csrank(g[1])
            b["p"] = self._csrank(g[1] - g[2])
            b["q"] = self._csrank(g[3] - g[4])
            b["r"] = self._csrank(g[1] - g[3])
            b["s"] = self._csrank(g[5])
        elif feat_type == "full":
            # All 6 features + all pairwise differences
            for i in range(1, 7):
                b[f"f{i}"] = self._csrank(g[i])
            for i in range(1, 7):
                for j in range(i+1, 7):
                    b[f"d{i}{j}"] = self._csrank(g[i] - g[j])
        elif feat_type == "minimal":
            # Just the core features that seem to work
            b["f5"] = self._csrank(g[5])
            b["core"] = self._csrank((g[1] + g[2] + g[3] + g[4]) / 4.0)
            b["f6"] = self._csrank(g[6])
        elif feat_type == "f5_focused":
            b["f5"] = self._csrank(g[5])
            b["f6"] = self._csrank(g[6])
            b["core"] = self._csrank((g[1] + g[2] + g[3] + g[4]) / 4.0)
            b["f1"] = self._csrank(g[1])
            b["d12"] = self._csrank(g[1] - g[2])
            b["d34"] = self._csrank(g[3] - g[4])
            b["d13"] = self._csrank(g[1] - g[3])
            b["f5_core"] = self._csrank(g[5] - (g[1] + g[2] + g[3] + g[4]) / 4.0)
        return b

    def _design(self, features):
        b = self._blocks(features)
        tickers = list(b[next(iter(b))].columns)
        X = np.hstack([d[tickers].values.reshape(-1, 1) for d in b.values()])
        return X, tickers, b[next(iter(b))].index

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
        pred_matrix = pd.DataFrame(
            raw.reshape(-1, len(tickers)), index=index, columns=tickers,
        )
        pred_matrix = pred_matrix.ewm(span=self.span, min_periods=1).mean()
        expanding_mean = pred_matrix.shift(1).expanding(min_periods=1).mean()
        pred_matrix = (pred_matrix - expanding_mean).fillna(0.0)
        pred_matrix = pred_matrix.sub(pred_matrix.mean(axis=1), axis=0)
        return pred_matrix.fillna(0.0)


# Test different configurations
configs = [
    # (name, span, depth, rounds, lambda, mcw, eta, features)
    ("demo_s480",         480, 2, 30, 5.0,  500, 0.05, "demo"),
    ("demo_s720",         720, 2, 30, 5.0,  500, 0.05, "demo"),
    ("full_s480",         480, 2, 30, 5.0,  500, 0.05, "full"),
    ("full_s720",         720, 2, 30, 5.0,  500, 0.05, "full"),
    ("f5_s480",           480, 2, 30, 5.0,  500, 0.05, "f5_focused"),
    ("f5_s720",           720, 2, 30, 5.0,  500, 0.05, "f5_focused"),
    ("minimal_s480",      480, 2, 30, 5.0,  500, 0.05, "minimal"),
    ("minimal_s720",      720, 2, 30, 5.0,  500, 0.05, "minimal"),
    ("full_d1_s480",      480, 1, 30, 5.0,  500, 0.05, "full"),
    ("full_d3_s480",      480, 3, 30, 5.0,  500, 0.05, "full"),
    ("full_r50_s480",     480, 2, 50, 5.0,  500, 0.05, "full"),
    ("full_l10_s480",     480, 2, 30, 10.0, 500, 0.05, "full"),
    ("full_mcw1k_s480",   480, 2, 30, 5.0, 1000, 0.05, "full"),
    ("demo_d1_s480",      480, 1, 50, 10.0, 1000, 0.03, "demo"),
]

for name, span, depth, rounds, lam, mcw, eta, feats in configs:
    cls = type(name, (TunedPredictor,), {
        "SPAN": span, "DEPTH": depth, "ROUNDS": rounds,
        "LAMBDA": lam, "MCW": mcw, "ETA": eta, "FEATURES": feats
    })
    res = walkforward_full(cls, "data/train", "small", verbose=False)
    print(f"{name:25s}: Sharpe={res['sharpe']:+.6f}")
