"""Test hybrid Ridge across all periods and compare with primary formula."""
import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
import numpy as np
import pandas as pd
from hybrid_ridge import HybridRidgePredictor
from data_loader import list_periods, load_period
from evaluation import backtest
from walkforward import compute_val_rows, walkforward_full

periods = sorted(list_periods("data/train"))

print("PER-PERIOD ANALYSIS: Hybrid Ridge")
print("=" * 70)
print(f"{'Period':>6} {'Sharpe':>10} {'IC_Tgt':>10} {'Turnover':>10}")
print("=" * 70)

sharpes = []
ics = []

for p in periods:
    d = load_period(p, "data/train", "small")
    n = len(d["features"])
    n_val = compute_val_rows(n)
    n_train = n - n_val
    
    train_feat = d["features"].iloc[:n_train]
    train_tgt = d["target"].iloc[:n_train]
    val_feat = d["features"].iloc[n_train:]
    val_ret = d["returns"].iloc[n_train:]
    val_tgt = d["target"].iloc[n_train:]
    
    pred_obj = HybridRidgePredictor()
    pred_obj.train(train_feat, train_tgt)
    pred = pred_obj.predict(val_feat)
    
    pf_ret = backtest(pred, val_ret)
    sh = float(pf_ret.mean() / pf_ret.std()) if pf_ret.std() > 0 else 0.0
    ic = np.nanmean(val_tgt.corrwith(pred, axis=1))
    turn = pred.shift(1).fillna(0).diff().abs().sum(axis=1).mean()
    
    sharpes.append(sh)
    ics.append(ic)
    print(f"{p:>6} {sh:>10.4f} {ic:>10.4f} {turn:>10.4f}")

print("=" * 70)
print(f"{'MEAN':>6} {np.mean(sharpes):>10.4f} {np.mean(ics):>10.4f}")
print(f"{'STD':>6} {np.std(sharpes):>10.4f}")
print(f"{'MIN':>6} {np.min(sharpes):>10.4f}")
neg = sum(1 for s in sharpes if s < 0)
print(f"Neg periods: {neg}/{len(sharpes)} ({100*neg/len(sharpes):.1f}%)")

# Full walkforward
print("\n\nFULL WALKFORWARD:")
res = walkforward_full(HybridRidgePredictor, gauge_fix=False, verbose=False)
print(f"Hybrid Ridge Full Sharpe: {res['sharpe']:.6f}")

# Test with different alphas and spans
print("\n\nHYPERPARAMETER SWEEP:")
for alpha in [100, 500, 1000, 5000, 10000]:
    for span in [72, 120, 240]:
        class T(HybridRidgePredictor):
            def __init__(self):
                super().__init__()
                self.alpha = alpha
                self.span = span
        res = walkforward_full(T, gauge_fix=False, verbose=False)
        print(f"  alpha={alpha:>6}, span={span:>4}: Full Sharpe = {res['sharpe']:.6f}")
