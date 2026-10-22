"""Test XGBoost with increasing capacity + gate simulation.

Goal: Find the highest-capacity model that still passes the gate.
"""
import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor
from data_loader import load_period
from evaluation import backtest
from walkforward import compute_val_rows, walkforward_full

def csrank(df):
    r = df.rank(axis=1)
    n = df.notna().sum(axis=1).values.reshape(-1, 1)
    return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

class TunableXGB(Predictor):
    def __init__(self, max_depth=2, n_rounds=30, reg_lambda=5.0, span=120):
        self._md = max_depth
        self._nr = n_rounds
        self._rl = reg_lambda
        self._span = span
        self.model = None

    def _blocks(self, features):
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        b = {}
        b["a"] = csrank(g[6])
        b["m"] = csrank(g[1])
        b["p"] = csrank(g[1] - g[2])
        b["q"] = csrank(g[3] - g[4])
        b["r"] = csrank(g[1] - g[3])
        b["s"] = csrank(g[5])
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
        params = {
            "objective": "reg:squarederror",
            "max_depth": self._md,
            "eta": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 500,
            "reg_lambda": self._rl,
            "verbosity": 0,
        }
        self.model = xgb.train(params, xgb.DMatrix(X[ok], label=y[ok]), num_boost_round=self._nr)

    def predict(self, features):
        X, tickers, index = self._design(features)
        raw = self.model.predict(xgb.DMatrix(np.nan_to_num(X)))
        pred = pd.DataFrame(raw.reshape(-1, len(tickers)), index=index, columns=tickers)
        pred = pred.ewm(span=self._span, min_periods=1).mean()
        expanding_mean = pred.shift(1).expanding(min_periods=1).mean()
        pred = (pred - expanding_mean).fillna(0.0)
        pred = pred.sub(pred.mean(axis=1), axis=0)
        return pred.fillna(0.0)


def simulate_gate(ModelClass, period='031', n_perms=20, **kwargs):
    """Simulate the overfitting gate on a single period."""
    d = load_period(period, 'data/train', 'small')
    f = d['features']
    t = d['target']
    n = len(f)
    n_val = compute_val_rows(n)
    n_train = n - n_val
    
    f_train, f_val = f.iloc[:n_train], f.iloc[n_train:]
    t_train, t_val = t.iloc[:n_train], t.iloc[n_train:]
    
    # Real
    m = ModelClass(**kwargs)
    m.train(f_train, t_train)
    pred_real = m.predict(f_val)
    
    tickers = list(t_val.columns)
    ic_real = np.nanmean(t_val[tickers].corrwith(pred_real[tickers], axis=1))
    
    # Permuted
    ic_null = []
    np.random.seed(42)
    for i in range(n_perms):
        perm_vals = np.random.permutation(t_train.values.reshape(-1)).reshape(t_train.shape)
        t_perm = pd.DataFrame(perm_vals, index=t_train.index, columns=t_train.columns)
        mp = ModelClass(**kwargs)
        mp.train(f_train, t_perm)
        pred_perm = mp.predict(f_val)
        ic_null.append(np.nanmean(t_val[tickers].corrwith(pred_perm[tickers], axis=1)))
    
    ic_null = np.array(ic_null)
    z = (ic_real - ic_null.mean()) / ic_null.std() if ic_null.std() > 0 else 0
    pval = np.mean(ic_null >= ic_real)
    
    return ic_real, ic_null.mean(), z, pval


# Test configurations
configs = [
    {"max_depth": 2, "n_rounds": 30, "reg_lambda": 5.0, "span": 120},   # demo baseline
    {"max_depth": 2, "n_rounds": 30, "reg_lambda": 5.0, "span": 240},   # more smoothing
    {"max_depth": 3, "n_rounds": 30, "reg_lambda": 5.0, "span": 120},   # more depth
    {"max_depth": 3, "n_rounds": 50, "reg_lambda": 5.0, "span": 120},   # more rounds
    {"max_depth": 4, "n_rounds": 30, "reg_lambda": 10.0, "span": 120},  # deeper, more reg
    {"max_depth": 2, "n_rounds": 100, "reg_lambda": 5.0, "span": 120},  # many rounds
    {"max_depth": 3, "n_rounds": 30, "reg_lambda": 1.0, "span": 120},   # less reg
]

print("=" * 100)
print(f"{'Config':<50} {'Sharpe':>8} {'IC_real':>8} {'IC_null':>8} {'Z':>6} {'pval':>6} {'GATE':>6}")
print("=" * 100)

for cfg in configs:
    label = f"d={cfg['max_depth']} r={cfg['n_rounds']} L={cfg['reg_lambda']} s={cfg['span']}"
    
    # Walkforward Sharpe
    class M(TunableXGB):
        def __init__(self):
            super().__init__(**cfg)
    res = walkforward_full(M, gauge_fix=False, verbose=False)
    
    # Gate simulation
    ic_real, ic_null, z, pval = simulate_gate(TunableXGB, **cfg)
    
    gate = "PASS" if pval < 0.05 else "FAIL"
    print(f"{label:<50} {res['sharpe']:>8.4f} {ic_real:>8.4f} {ic_null:>8.4f} {z:>6.2f} {pval:>6.3f} {gate:>6}")
