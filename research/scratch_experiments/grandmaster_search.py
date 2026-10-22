"""GRANDMASTER SIGNAL SEARCH: Find the signal with highest CONSISTENT IC and Sharpe.

Tests every possible simple signal across all 31 training periods.
The goal is to find the signal that Rank 1 is likely using.
"""
import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
import numpy as np
import pandas as pd
from data_loader import list_periods, load_period
from evaluation import backtest
from walkforward import compute_val_rows

periods = sorted(list_periods("data/train"))

def csrank(df):
    r = df.rank(axis=1)
    n = df.notna().sum(axis=1).values.reshape(-1, 1)
    return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

def eval_signal(pred_fn, span=120, tilt_removal=True):
    """Evaluate a signal across all 31 periods. Returns per-period stats."""
    sharpes = []
    ics = []
    ret_ics = []
    for p in periods:
        d = load_period(p, "data/train", "small")
        n = len(d["features"])
        n_val = compute_val_rows(n)
        n_train = n - n_val
        
        # Generate raw predictions on validation set
        raw = pred_fn(d["features"].iloc[n_train:])
        
        # Apply EWM smoothing
        pred = raw.ewm(span=span, min_periods=1).mean()
        
        # Causal tilt removal
        if tilt_removal:
            expanding_mean = pred.shift(1).expanding(min_periods=1).mean()
            pred = (pred - expanding_mean).fillna(0.0)
        
        # De-mean
        pred = pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)
        
        val_ret = d["returns"].iloc[n_train:]
        val_tgt = d["target"].iloc[n_train:]
        
        pf_ret = backtest(pred, val_ret)
        sh = float(pf_ret.mean() / pf_ret.std()) if pf_ret.std() > 0 else 0.0
        ic = np.nanmean(val_tgt.corrwith(pred, axis=1))
        ret_ic = np.nanmean(val_ret.corrwith(pred, axis=1))
        
        sharpes.append(sh)
        ics.append(ic)
        ret_ics.append(ret_ic)
    
    return np.array(sharpes), np.array(ics), np.array(ret_ics)

# ============================================================
# DEFINE ALL SIGNALS
# ============================================================
signals = {}

# 1. Individual features (positive and negative)
for i in range(1, 7):
    for sign, label in [(1, f"+F{i}"), (-1, f"-F{i}")]:
        def make_pred(sign=sign, i=i):
            def pred_fn(features):
                return sign * csrank(features[f"Feature.{i}"])
            return pred_fn
        signals[label] = make_pred()

# 2. Core and negative core
def make_core(sign=1):
    def pred_fn(features):
        g = {j: features[f"Feature.{j}"] for j in range(1, 7)}
        core = sum(csrank(g[j]) for j in range(1, 5)) / 4.0
        return sign * core
    return pred_fn
signals["core"] = make_core(1)
signals["-core"] = make_core(-1)

# 3. Primary formula variants
def make_primary():
    def pred_fn(features):
        g = {j: features[f"Feature.{j}"] for j in range(1, 7)}
        ranks = {j: csrank(g[j]) for j in range(1, 7)}
        core = sum(ranks[j] for j in range(1, 5)) / 4.0
        return -(core**3) + core * ranks[6]
    return pred_fn
signals["primary"] = make_primary()

def make_neg_primary():
    def pred_fn(features):
        g = {j: features[f"Feature.{j}"] for j in range(1, 7)}
        ranks = {j: csrank(g[j]) for j in range(1, 7)}
        core = sum(ranks[j] for j in range(1, 5)) / 4.0
        return (core**3) - core * ranks[6]
    return pred_fn
signals["-primary"] = make_neg_primary()

# 4. Pairwise differences
for i in range(1, 7):
    for j in range(i+1, 7):
        for sign, label in [(1, f"F{i}-F{j}"), (-1, f"F{j}-F{i}")]:
            def make_diff(i=i, j=j, sign=sign):
                def pred_fn(features):
                    return sign * csrank(features[f"Feature.{i}"] - features[f"Feature.{j}"])
                return pred_fn
            signals[label] = make_diff()

# 5. core * F5, core * F6
for i in [5, 6]:
    def make_core_fi(i=i):
        def pred_fn(features):
            g = {j: features[f"Feature.{j}"] for j in range(1, 7)}
            core = sum(csrank(g[j]) for j in range(1, 5)) / 4.0
            return core * csrank(g[i])
        return pred_fn
    signals[f"core*F{i}"] = make_core_fi()
    def make_neg_core_fi(i=i):
        def pred_fn(features):
            g = {j: features[f"Feature.{j}"] for j in range(1, 7)}
            core = sum(csrank(g[j]) for j in range(1, 5)) / 4.0
            return -(core * csrank(g[i]))
        return pred_fn
    signals[f"-core*F{i}"] = make_neg_core_fi()

# 6. negcore3
def make_negcore3():
    def pred_fn(features):
        g = {j: features[f"Feature.{j}"] for j in range(1, 7)}
        core = sum(csrank(g[j]) for j in range(1, 5)) / 4.0
        return -(core**3)
    return pred_fn
signals["negcore3"] = make_negcore3()

# 7. Just -F6 with power concentration
def make_neg_f6_power():
    def pred_fn(features):
        r = -csrank(features["Feature.6"])
        return np.sign(r) * np.abs(r)**3
    return pred_fn
signals["-F6^3"] = make_neg_f6_power()

# 8. Equal-weight blend of -F1, -F2, ..., -F6
def make_neg_all():
    def pred_fn(features):
        return sum(-csrank(features[f"Feature.{j}"]) for j in range(1, 7)) / 6.0
    return pred_fn
signals["-allF"] = make_neg_all()

print(f"Testing {len(signals)} signals across {len(periods)} periods (span=120, tilt=True)...")
print("=" * 130)
print(f"{'Signal':<20} {'MeanSh':>8} {'MedSh':>8} {'MinSh':>8} {'MaxSh':>8} {'StdSh':>8} {'Neg%':>6} {'MeanIC':>8} {'MedIC':>8} {'RetIC':>8}")
print("=" * 130)

results = []
for name, pred_fn in signals.items():
    sharpes, ics, ret_ics = eval_signal(pred_fn, span=120, tilt_removal=True)
    neg_pct = 100 * np.mean(sharpes < 0)
    results.append({
        "name": name,
        "mean_sh": np.mean(sharpes), "med_sh": np.median(sharpes),
        "min_sh": np.min(sharpes), "max_sh": np.max(sharpes),
        "std_sh": np.std(sharpes), "neg_pct": neg_pct,
        "mean_ic": np.mean(ics), "med_ic": np.median(ics),
        "ret_ic": np.mean(ret_ics),
    })
    
# Sort by mean Sharpe
results.sort(key=lambda x: x["mean_sh"], reverse=True)

for r in results:
    print(f"{r['name']:<20} {r['mean_sh']:>8.4f} {r['med_sh']:>8.4f} {r['min_sh']:>8.4f} {r['max_sh']:>8.4f} {r['std_sh']:>8.4f} {r['neg_pct']:>5.1f}% {r['mean_ic']:>8.4f} {r['med_ic']:>8.4f} {r['ret_ic']:>8.4f}")

print("\n\n=== TOP 10 BY CONSISTENCY (lowest neg%) ===")
results_by_neg = sorted(results, key=lambda x: (x["neg_pct"], -x["mean_sh"]))
for r in results_by_neg[:10]:
    print(f"{r['name']:<20} {r['mean_sh']:>8.4f} {r['neg_pct']:>5.1f}% {r['mean_ic']:>8.4f} {r['ret_ic']:>8.4f}")

print("\n\n=== TOP 10 BY RETURN IC (most correlated with actual returns) ===")
results_by_ret_ic = sorted(results, key=lambda x: x["ret_ic"], reverse=True)
for r in results_by_ret_ic[:10]:
    print(f"{r['name']:<20} {r['mean_sh']:>8.4f} {r['neg_pct']:>5.1f}% {r['mean_ic']:>8.4f} {r['ret_ic']:>8.4f}")
