"""Comprehensive last-period sweep: test every promising configuration.
The server trains on the latest period, so last-period validation is the best proxy."""
import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")

import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor
from walkforward import walkforward_backtest

# =================== CONFIG 1: Demo exact (baseline) ===================
class Config1(Predictor):
    """Demo exact baseline, span=120"""
    def __init__(self):
        self.model = None
        self.params = {"objective": "reg:squarederror", "max_depth": 2, "eta": 0.05,
                       "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 500,
                       "reg_lambda": 5.0, "verbosity": 0, "nthread": 4}
        self.n_rounds = 30; self.span = 120
    @staticmethod
    def _csrank(df):
        r = df.rank(axis=1); n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))
    def _design(self, features):
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        b = {"a": self._csrank(g[6]), "m": self._csrank(g[1]),
             "p": self._csrank(g[1] - g[2]), "q": self._csrank(g[3] - g[4]),
             "r": self._csrank(g[1] - g[3]), "s": self._csrank(g[5])}
        tickers = list(b["a"].columns)
        X = np.hstack([d[tickers].values.reshape(-1, 1) for d in b.values()])
        return X, tickers, b["a"].index
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

# =================== CONFIG 2: Enhanced + primary formula features, span=720 ===================
class Config2(Predictor):
    """Demo features + primary & negcore3, span=720"""
    def __init__(self):
        self.model = None
        self.params = {"objective": "reg:squarederror", "max_depth": 2, "eta": 0.05,
                       "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 500,
                       "reg_lambda": 5.0, "verbosity": 0, "nthread": 4}
        self.n_rounds = 30; self.span = 720
    @staticmethod
    def _csrank(df):
        r = df.rank(axis=1); n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))
    def _design(self, features):
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        tickers = sorted(g[1].columns, key=lambda name: int(name.split(".")[-1]))
        ranks = []
        for i in range(1, 7):
            v = np.nan_to_num(g[i][tickers].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
            o = np.argsort(np.argsort(v, axis=1, kind="mergesort"), axis=1, kind="mergesort")
            ranks.append((o.astype(float) - 9.5) / 9.5)
        core = sum(ranks[:4]) / 4.0
        primary = -core**3 + core * ranks[5]
        negcore3 = -core**3
        b = {"a": self._csrank(g[6]), "m": self._csrank(g[1]),
             "p": self._csrank(g[1] - g[2]), "q": self._csrank(g[3] - g[4]),
             "r": self._csrank(g[1] - g[3]), "s": self._csrank(g[5]),
             "primary": pd.DataFrame(primary, index=g[1].index, columns=tickers),
             "negcore3": pd.DataFrame(negcore3, index=g[1].index, columns=tickers)}
        X = np.hstack([d[tickers].values.reshape(-1, 1) for d in b.values()])
        return X, tickers, b["a"].index
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

# =================== CONFIG 3: Ultra-constrained XGB, fewer rounds, ultra-smooth ===================
class Config3(Config2):
    """Same as Config2 but 15 rounds, span=1440"""
    def __init__(self):
        super().__init__()
        self.n_rounds = 15; self.span = 1440

# =================== CONFIG 4: XGB with ONLY primary+negcore3, span=720 ===================
class Config4(Predictor):
    """Only primary & negcore3 features, span=720"""
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
        primary = -core**3 + core * ranks[5]
        negcore3 = -core**3
        X = np.column_stack([primary.reshape(-1), negcore3.reshape(-1)])
        return X, tickers, features.index
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

# =================== CONFIG 5: Ridge on primary, span=720 ===================
class Config5(Predictor):
    """Ridge regression on primary only, span=720"""
    def __init__(self):
        self.beta = None; self.span = 720; self.reg = 5000.0
    def _design(self, features):
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        tickers = sorted(g[1].columns, key=lambda name: int(name.split(".")[-1]))
        ranks = []
        for i in range(1, 7):
            v = np.nan_to_num(g[i][tickers].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
            o = np.argsort(np.argsort(v, axis=1, kind="mergesort"), axis=1, kind="mergesort")
            ranks.append((o.astype(float) - 9.5) / 9.5)
        core = sum(ranks[:4]) / 4.0
        primary = -core**3 + core * ranks[5]
        X = primary.reshape(-1, 1)
        return X, tickers, features.index
    def train(self, features, target):
        X, tickers, _ = self._design(features)
        y = target[tickers].values.reshape(-1)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        gram = X[ok].T @ X[ok]; rhs = X[ok].T @ y[ok]
        self.beta = np.linalg.solve(gram + self.reg * np.eye(1), rhs)
    def predict(self, features):
        X, tickers, index = self._design(features)
        raw = X @ self.beta
        pred = pd.DataFrame(raw.reshape(-1, len(tickers)), index=index, columns=tickers)
        pred = pred.ewm(span=self.span, min_periods=1).mean()
        em = pred.shift(1).expanding(min_periods=1).mean()
        pred = (pred - em).fillna(0.0)
        return pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)

# =================== CONFIG 6: XGB no tilt removal, span=720 ===================
class Config6(Config2):
    """Same as Config2 but WITHOUT tilt removal"""
    def predict(self, features):
        X, tickers, index = self._design(features)
        raw = self.model.predict(xgb.DMatrix(np.nan_to_num(X)))
        pred = pd.DataFrame(raw.reshape(-1, len(tickers)), index=index, columns=tickers)
        pred = pred.ewm(span=self.span, min_periods=1).mean()
        return pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)

# =================== CONFIG 7: Very low eta, more rounds ===================
class Config7(Predictor):
    """Very low learning rate, more rounds, span=720"""
    def __init__(self):
        self.model = None
        self.params = {"objective": "reg:squarederror", "max_depth": 2, "eta": 0.01,
                       "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 500,
                       "reg_lambda": 10.0, "verbosity": 0, "nthread": 4}
        self.n_rounds = 50; self.span = 720
    @staticmethod
    def _csrank(df):
        r = df.rank(axis=1); n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))
    def _design(self, features):
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        tickers = sorted(g[1].columns, key=lambda name: int(name.split(".")[-1]))
        ranks = []
        for i in range(1, 7):
            v = np.nan_to_num(g[i][tickers].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
            o = np.argsort(np.argsort(v, axis=1, kind="mergesort"), axis=1, kind="mergesort")
            ranks.append((o.astype(float) - 9.5) / 9.5)
        core = sum(ranks[:4]) / 4.0
        primary = -core**3 + core * ranks[5]
        negcore3 = -core**3
        b = {"a": self._csrank(g[6]), "m": self._csrank(g[1]),
             "p": self._csrank(g[1] - g[2]), "q": self._csrank(g[3] - g[4]),
             "r": self._csrank(g[1] - g[3]), "s": self._csrank(g[5]),
             "primary": pd.DataFrame(primary, index=g[1].index, columns=tickers),
             "negcore3": pd.DataFrame(negcore3, index=g[1].index, columns=tickers)}
        X = np.hstack([d[tickers].values.reshape(-1, 1) for d in b.values()])
        return X, tickers, b["a"].index
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

# =================== CONFIG 8: XGB depth=1, span=720 ===================
class Config8(Config2):
    """Depth=1 (stumps only), span=720"""
    def __init__(self):
        super().__init__()
        self.params["max_depth"] = 1

# Run all
configs = [
    ("C1: Demo baseline span=120", Config1),
    ("C2: Enhanced+primary span=720", Config2),
    ("C3: Enhanced 15rnd span=1440", Config3),
    ("C4: PrimaryOnly span=720", Config4),
    ("C5: Ridge primary span=720", Config5),
    ("C6: Enhanced NO-TILT span=720", Config6),
    ("C7: LowEta enhanced span=720", Config7),
    ("C8: Depth1 enhanced span=720", Config8),
]

print("="*70)
print("LAST-PERIOD VALIDATION SHARPE (proxy for server performance)")
print("="*70)

for name, cls in configs:
    try:
        res = walkforward_backtest(cls, verbose=False)
        print(f"{name:38s} Sharpe={res['sharpe']:+.6f}")
    except Exception as e:
        print(f"{name:38s} ERROR: {e}")
