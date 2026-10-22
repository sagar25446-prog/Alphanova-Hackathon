"""Full walkforward of top configs to verify robustness across all periods."""
import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")

import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor
from walkforward import walkforward_full

# ============================================================
# Config A: enhanced depth=3, span=720  (BEST single-period)
# ============================================================
class ConfigA(Predictor):
    def __init__(self):
        self.model = None
        self.params = {"objective": "reg:squarederror", "max_depth": 3, "eta": 0.05,
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
        b = {"a": self._csrank(g[6]), "m": self._csrank(g[1]),
             "p": self._csrank(g[1] - g[2]), "q": self._csrank(g[3] - g[4]),
             "r": self._csrank(g[1] - g[3]), "s": self._csrank(g[5]),
             "primary": pd.DataFrame(-core**3 + core * ranks[5], index=g[1].index, columns=tickers),
             "negcore3": pd.DataFrame(-core**3, index=g[1].index, columns=tickers)}
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

# ============================================================
# Config B: enhanced eta=0.1, 20 rounds, span=720
# ============================================================
class ConfigB(ConfigA):
    def __init__(self):
        super().__init__()
        self.params["max_depth"] = 2
        self.params["eta"] = 0.1
        self.n_rounds = 20

# ============================================================
# Config C: enhanced eta=0.03, 50 rounds, span=720
# ============================================================
class ConfigC(ConfigA):
    def __init__(self):
        super().__init__()
        self.params["max_depth"] = 2
        self.params["eta"] = 0.03
        self.n_rounds = 50

# ============================================================
# Config D: depth=3, eta=0.1, 15 rounds, span=720 (aggressive but fewer rounds)
# ============================================================
class ConfigD(ConfigA):
    def __init__(self):
        super().__init__()
        self.params["max_depth"] = 3
        self.params["eta"] = 0.1
        self.n_rounds = 15

# ============================================================
# Config E: depth=3, higher regularization to compensate
# ============================================================
class ConfigE(ConfigA):
    def __init__(self):
        super().__init__()
        self.params["max_depth"] = 3
        self.params["reg_lambda"] = 10.0
        self.params["min_child_weight"] = 1000

# ============================================================
# Config F: Original submitted (depth=2, eta=0.05, 30rnd) for comparison
# ============================================================
class ConfigF(ConfigA):
    def __init__(self):
        super().__init__()
        self.params["max_depth"] = 2

configs = [
    ("A: depth=3 eta=0.05 30rnd", ConfigA),
    ("B: depth=2 eta=0.1 20rnd", ConfigB),
    ("C: depth=2 eta=0.03 50rnd", ConfigC),
    ("D: depth=3 eta=0.1 15rnd", ConfigD),
    ("E: depth=3 lam=10 mcw=1000", ConfigE),
    ("F: SUBMITTED (depth=2 baseline)", ConfigF),
]

print("=" * 75)
print("FULL WALKFORWARD (31 periods) - ROBUSTNESS CHECK")
print("=" * 75)

for name, cls in configs:
    try:
        res = walkforward_full(cls, gauge_fix=False, verbose=False)
        sharpe = res['sharpe']
        # Count positive/negative periods
        pp = res.get('per_period', {})
        pos = sum(1 for v in pp.values() if v['sharpe'] > 0)
        neg = sum(1 for v in pp.values() if v['sharpe'] <= 0)
        # Recent 9 periods average
        recent_keys = sorted(pp.keys())[-9:]
        recent_avg = np.mean([pp[k]['sharpe'] for k in recent_keys])
        print(f"{name:35s} Full={sharpe:+.6f}  Recent9={recent_avg:+.6f}  Win={pos}/{pos+neg}")
    except Exception as e:
        print(f"{name:35s} ERROR: {e}")
