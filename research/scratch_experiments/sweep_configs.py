"""Full walkforward sweep: test different XGBoost feature sets and smoothing spans."""
import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")

import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor
from walkforward import walkforward_full

# ===== Config 1: demo_engineered EXACT (baseline) =====
class DemoExact(Predictor):
    def __init__(self):
        self.model = None
        self.params = {"objective": "reg:squarederror", "max_depth": 2, "eta": 0.05,
                       "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 500,
                       "reg_lambda": 5.0, "verbosity": 0, "nthread": 4}
        self.n_rounds = 30
        self.span = 120

    @staticmethod
    def _csrank(df):
        r = df.rank(axis=1)
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

    def _blocks(self, features):
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        b = {}
        b["a"] = self._csrank(g[6])
        b["m"] = self._csrank(g[1])
        b["p"] = self._csrank(g[1] - g[2])
        b["q"] = self._csrank(g[3] - g[4])
        b["r"] = self._csrank(g[1] - g[3])
        b["s"] = self._csrank(g[5])
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


# ===== Config 2: Enhanced features with primary formula as XGB feature =====
class EnhancedXGB(Predictor):
    def __init__(self):
        self.model = None
        self.params = {"objective": "reg:squarederror", "max_depth": 2, "eta": 0.05,
                       "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 500,
                       "reg_lambda": 5.0, "verbosity": 0, "nthread": 4}
        self.n_rounds = 30
        self.span = 120

    @staticmethod
    def _csrank(df):
        r = df.rank(axis=1)
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

    def _np_csrank(self, arr):
        order = np.argsort(np.argsort(arr, axis=1, kind="mergesort"), axis=1, kind="mergesort")
        return (order.astype(float) - 9.5) / 9.5

    def _blocks(self, features):
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        tickers = sorted(g[1].columns, key=lambda name: int(name.split(".")[-1]))
        
        # Original demo features
        b = {}
        b["a"] = self._csrank(g[6])
        b["m"] = self._csrank(g[1])
        b["p"] = self._csrank(g[1] - g[2])
        b["q"] = self._csrank(g[3] - g[4])
        b["r"] = self._csrank(g[1] - g[3])
        b["s"] = self._csrank(g[5])
        
        # THE KEY: primary formula as a feature for XGB
        # -core^3 + core*rank6 has IC=+0.074 — the strongest feature known
        ranks = []
        for i in range(1, 7):
            vals = np.nan_to_num(g[i][tickers].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
            order = np.argsort(np.argsort(vals, axis=1, kind="mergesort"), axis=1, kind="mergesort")
            ranks.append((order.astype(float) - 9.5) / 9.5)
        
        core = sum(ranks[:4]) / 4.0
        primary = -core**3 + core * ranks[5]
        neg_core_cubed = -core**3
        
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


# ===== Config 3: Enhanced + more smoothing =====
class EnhancedSmooth(EnhancedXGB):
    def __init__(self):
        super().__init__()
        self.span = 480


# ===== Config 4: Enhanced + even more smoothing =====
class EnhancedSmoother(EnhancedXGB):
    def __init__(self):
        super().__init__()
        self.span = 720


# ===== Config 5: Primary-only features (minimal) =====
class PrimaryOnlyXGB(Predictor):
    def __init__(self):
        self.model = None
        self.params = {"objective": "reg:squarederror", "max_depth": 2, "eta": 0.05,
                       "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 500,
                       "reg_lambda": 5.0, "verbosity": 0, "nthread": 4}
        self.n_rounds = 30
        self.span = 480

    @staticmethod
    def _csrank(df):
        r = df.rank(axis=1)
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

    def _design(self, features):
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        tickers = sorted(g[1].columns, key=lambda name: int(name.split(".")[-1]))
        
        ranks = []
        for i in range(1, 7):
            vals = np.nan_to_num(g[i][tickers].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
            order = np.argsort(np.argsort(vals, axis=1, kind="mergesort"), axis=1, kind="mergesort")
            ranks.append((order.astype(float) - 9.5) / 9.5)
        
        core = sum(ranks[:4]) / 4.0
        primary = -core**3 + core * ranks[5]
        neg_core3 = -core**3
        core_r6 = core * ranks[5]
        
        X = np.column_stack([
            primary.reshape(-1),
            neg_core3.reshape(-1),
            core_r6.reshape(-1),
            self._csrank(g[6])[tickers].values.reshape(-1),
            self._csrank(g[1])[tickers].values.reshape(-1),
        ])
        return X, tickers, g[1].index

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


# ===== Config 6: Demo baseline with span=480 =====
class DemoSmooth(DemoExact):
    def __init__(self):
        super().__init__()
        self.span = 480


# Run all configs
configs = [
    ("DemoExact (span=120)", DemoExact),
    ("DemoSmooth (span=480)", DemoSmooth),
    ("EnhancedXGB (span=120)", EnhancedXGB),
    ("EnhancedSmooth (span=480)", EnhancedSmooth),
    ("EnhancedSmoother (span=720)", EnhancedSmoother),
    ("PrimaryOnlyXGB (span=480)", PrimaryOnlyXGB),
]

print("="*70)
print("FULL WALKFORWARD SHARPE SWEEP")
print("="*70)

for name, cls in configs:
    try:
        res = walkforward_full(cls, verbose=False)
        sharpe = res['sharpe']
        ic = res.get('ic', 'N/A')
        print(f"{name:40s} Sharpe={sharpe:+.6f}  IC={ic}")
    except Exception as e:
        print(f"{name:40s} ERROR: {e}")
