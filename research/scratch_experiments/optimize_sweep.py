"""Systematic optimization sweep across recent periods (025-031).
Tests: feature combos, spans, XGB hyperparams, determinism, clipping."""
import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")

import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor
from data_loader import list_periods, load_period

DATA_DIR = "data/train"
UNIVERSE = "small"

# ============================================================
# Helper: run single-period train/predict/sharpe
# ============================================================
def eval_period(cls, period_id, data_dir=DATA_DIR, universe=UNIVERSE):
    """Train on train split, predict on val split, return Sharpe."""
    periods = list_periods(data_dir, universe)
    data = load_period(period_id, data_dir, universe)
    features = data["features"]
    target = data["target"]
    returns = data["returns"]
    
    n_total = len(features.index)
    n_val = min(800, int(0.2 * n_total))
    n_train = n_total - n_val
    
    train_idx = features.index[:n_train]
    val_idx = features.index[n_train:]
    
    # Build train/val feature dicts
    train_features = {}
    val_features = {}
    for key in features.keys():
        train_features[key] = features[key].iloc[:n_train]
        val_features[key] = features[key].iloc[n_train:]
    
    # Convert to proper format
    train_features_obj = type('Features', (), {'__getitem__': lambda s, k: train_features[k], 'keys': lambda s: train_features.keys(), 'index': train_idx})()
    val_features_obj = type('Features', (), {'__getitem__': lambda s, k: val_features[k], 'keys': lambda s: val_features.keys(), 'index': val_idx})()
    
    # Wrap in pd-like
    class FeatDict(dict):
        def __init__(self, d, idx):
            super().__init__(d)
            self._index = idx
        @property
        def index(self):
            return self._index
    
    train_f = FeatDict(train_features, train_idx)
    val_f = FeatDict(val_features, val_idx)
    
    train_target = target.iloc[:n_train]
    val_returns = returns.iloc[n_train:]
    
    model = cls()
    model.train(train_f, train_target)
    preds = model.predict(val_f)
    
    # Compute Sharpe
    tickers = sorted(preds.columns, key=lambda name: int(name.split(".")[-1]))
    pf_ret = (preds[tickers].shift(1) * val_returns[tickers]).sum(axis=1)
    # Transaction costs
    turnover = preds[tickers].diff().abs().sum(axis=1)
    pf_ret_net = pf_ret - 0.0005 * turnover
    pf_ret_net = pf_ret_net.iloc[1:]  # drop first NaN
    
    sharpe = pf_ret_net.mean() / pf_ret_net.std() if pf_ret_net.std() > 0 else 0
    return sharpe

# ============================================================
# Use walkforward_backtest for proper evaluation
# ============================================================
from walkforward import walkforward_backtest, walkforward_full

def eval_recent_periods(cls, n_recent=7):
    """Evaluate on most recent n_recent periods using full walkforward."""
    try:
        res = walkforward_full(cls, verbose=False)
        return res['sharpe'], res.get('per_period_sharpes', [])
    except Exception as e:
        return None, str(e)

# ============================================================
# Base class factory
# ============================================================
def make_predictor(span=720, eta=0.05, n_rounds=30, max_depth=2,
                   min_child_weight=500, reg_lambda=5.0,
                   subsample=0.8, colsample=0.8,
                   features_mode="enhanced", clip=None,
                   no_tilt=False):
    
    class CustomPredictor(Predictor):
        def __init__(self):
            self.model = None
            self.params = {
                "objective": "reg:squarederror",
                "max_depth": max_depth,
                "eta": eta,
                "subsample": subsample,
                "colsample_bytree": colsample,
                "min_child_weight": min_child_weight,
                "reg_lambda": reg_lambda,
                "verbosity": 0,
                "nthread": 4,
            }
            self._n_rounds = n_rounds
            self._span = span
            self._features_mode = features_mode
            self._clip = clip
            self._no_tilt = no_tilt

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
                v = np.nan_to_num(g[i][tickers].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
                o = np.argsort(np.argsort(v, axis=1, kind="mergesort"), axis=1, kind="mergesort")
                ranks.append((o.astype(float) - 9.5) / 9.5)
            
            core = sum(ranks[:4]) / 4.0
            primary = -core**3 + core * ranks[5]
            negcore3 = -core**3
            
            b = {}
            
            if self._features_mode == "demo":
                b["a"] = self._csrank(g[6])
                b["m"] = self._csrank(g[1])
                b["p"] = self._csrank(g[1] - g[2])
                b["q"] = self._csrank(g[3] - g[4])
                b["r"] = self._csrank(g[1] - g[3])
                b["s"] = self._csrank(g[5])
            elif self._features_mode == "enhanced":
                b["a"] = self._csrank(g[6])
                b["m"] = self._csrank(g[1])
                b["p"] = self._csrank(g[1] - g[2])
                b["q"] = self._csrank(g[3] - g[4])
                b["r"] = self._csrank(g[1] - g[3])
                b["s"] = self._csrank(g[5])
                b["primary"] = pd.DataFrame(primary, index=g[1].index, columns=tickers)
                b["negcore3"] = pd.DataFrame(negcore3, index=g[1].index, columns=tickers)
            elif self._features_mode == "enhanced_plus":
                b["a"] = self._csrank(g[6])
                b["m"] = self._csrank(g[1])
                b["p"] = self._csrank(g[1] - g[2])
                b["q"] = self._csrank(g[3] - g[4])
                b["r"] = self._csrank(g[1] - g[3])
                b["s"] = self._csrank(g[5])
                b["primary"] = pd.DataFrame(primary, index=g[1].index, columns=tickers)
                b["negcore3"] = pd.DataFrame(negcore3, index=g[1].index, columns=tickers)
                # Extra interactions
                b["core_r6"] = pd.DataFrame(core * ranks[5], index=g[1].index, columns=tickers)
                b["core_r1"] = pd.DataFrame(core * ranks[0], index=g[1].index, columns=tickers)
                b["r1_r6"] = pd.DataFrame(ranks[0] * ranks[5], index=g[1].index, columns=tickers)
            elif self._features_mode == "kitchen_sink":
                b["a"] = self._csrank(g[6])
                b["m"] = self._csrank(g[1])
                b["p"] = self._csrank(g[1] - g[2])
                b["q"] = self._csrank(g[3] - g[4])
                b["r"] = self._csrank(g[1] - g[3])
                b["s"] = self._csrank(g[5])
                b["primary"] = pd.DataFrame(primary, index=g[1].index, columns=tickers)
                b["negcore3"] = pd.DataFrame(negcore3, index=g[1].index, columns=tickers)
                b["core_r6"] = pd.DataFrame(core * ranks[5], index=g[1].index, columns=tickers)
                b["core_r1"] = pd.DataFrame(core * ranks[0], index=g[1].index, columns=tickers)
                b["r1_r6"] = pd.DataFrame(ranks[0] * ranks[5], index=g[1].index, columns=tickers)
                b["core_sq"] = pd.DataFrame(core**2, index=g[1].index, columns=tickers)
                b["r5_r6"] = pd.DataFrame(ranks[4] * ranks[5], index=g[1].index, columns=tickers)
                b["f2_f4_diff"] = self._csrank(g[2] - g[4])
            elif self._features_mode == "primary_only":
                b["primary"] = pd.DataFrame(primary, index=g[1].index, columns=tickers)
                b["negcore3"] = pd.DataFrame(negcore3, index=g[1].index, columns=tickers)
            
            X = np.hstack([d[tickers].values.reshape(-1, 1) for d in b.values()])
            idx = b[next(iter(b))].index if isinstance(b[next(iter(b))], pd.DataFrame) else b[next(iter(b))].index
            return X, tickers, idx

        def train(self, features, target):
            X, tickers, _ = self._design(features)
            y = target[tickers].values.reshape(-1)
            ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
            self.model = xgb.train(self.params, xgb.DMatrix(X[ok], label=y[ok]),
                                   num_boost_round=self._n_rounds)

        def predict(self, features):
            X, tickers, index = self._design(features)
            raw = self.model.predict(xgb.DMatrix(np.nan_to_num(X)))
            pred = pd.DataFrame(raw.reshape(-1, len(tickers)), index=index, columns=tickers)
            
            if self._clip is not None:
                pred = pred.clip(-self._clip, self._clip)
            
            pred = pred.ewm(span=self._span, min_periods=1).mean()
            
            if not self._no_tilt:
                em = pred.shift(1).expanding(min_periods=1).mean()
                pred = (pred - em).fillna(0.0)
            
            pred = pred.sub(pred.mean(axis=1), axis=0)
            return pred.fillna(0.0)
    
    return CustomPredictor

# ============================================================
# SWEEP
# ============================================================
configs = []

# Span sweep (with enhanced features)
for sp in [480, 600, 720, 960, 1200]:
    configs.append((f"enhanced span={sp}", 
                     make_predictor(span=sp, features_mode="enhanced")))

# Feature mode sweep (with span=720)
for fm in ["demo", "enhanced", "enhanced_plus", "kitchen_sink", "primary_only"]:
    configs.append((f"{fm} span=720",
                     make_predictor(span=720, features_mode=fm)))

# Deterministic (no subsampling)
configs.append(("enhanced det span=720",
                make_predictor(span=720, subsample=1.0, colsample=1.0)))

# Hyperparameter variations
configs.append(("enhanced eta=0.03 50rnd",
                make_predictor(span=720, eta=0.03, n_rounds=50)))
configs.append(("enhanced eta=0.1 20rnd",
                make_predictor(span=720, eta=0.1, n_rounds=20)))
configs.append(("enhanced lambda=10",
                make_predictor(span=720, reg_lambda=10.0)))
configs.append(("enhanced lambda=2",
                make_predictor(span=720, reg_lambda=2.0)))
configs.append(("enhanced mcw=200",
                make_predictor(span=720, min_child_weight=200)))
configs.append(("enhanced mcw=1000",
                make_predictor(span=720, min_child_weight=1000)))
configs.append(("enhanced depth=3",
                make_predictor(span=720, max_depth=3)))

# Clipping
configs.append(("enhanced clip=0.5",
                make_predictor(span=720, clip=0.5)))
configs.append(("enhanced clip=1.0",
                make_predictor(span=720, clip=1.0)))

# No tilt removal
configs.append(("enhanced no-tilt span=720",
                make_predictor(span=720, no_tilt=True)))

# Combos of best ideas
configs.append(("enhanced+ det span=960",
                make_predictor(span=960, features_mode="enhanced_plus", subsample=1.0, colsample=1.0)))
configs.append(("enhanced det span=960",
                make_predictor(span=960, subsample=1.0, colsample=1.0)))

print("=" * 70)
print("LAST-PERIOD VALIDATION SHARPE SWEEP")
print("=" * 70)

results = []
for name, cls in configs:
    try:
        res = walkforward_backtest(cls, verbose=False)
        sharpe = res['sharpe']
        results.append((name, sharpe))
        print(f"{name:40s} Sharpe={sharpe:+.6f}")
    except Exception as e:
        print(f"{name:40s} ERROR: {e}")

print("\n" + "=" * 70)
print("TOP 5 CONFIGS (sorted by Sharpe)")
print("=" * 70)
results.sort(key=lambda x: x[1], reverse=True)
for name, sharpe in results[:5]:
    print(f"{name:40s} Sharpe={sharpe:+.6f}")
