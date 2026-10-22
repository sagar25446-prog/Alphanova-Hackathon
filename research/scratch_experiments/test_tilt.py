"""Test: does causal tilt removal improve stability?"""
import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
import numpy as np
import pandas as pd
from predictor import Predictor
from data_loader import list_periods, load_period
from evaluation import backtest
from walkforward import compute_val_rows, walkforward_full

def csrank(df):
    r = df.rank(axis=1)
    n = df.notna().sum(axis=1).values.reshape(-1, 1)
    return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

class PrimaryNoTilt(Predictor):
    """Primary with causal tilt removal (like demo_engineered)."""
    def __init__(self):
        self.span = 72

    def train(self, features, target):
        pass

    def predict(self, features):
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        ranks = [csrank(g[i]) for i in range(1, 7)]
        core = sum(ranks[:4]) / 4.0
        primary = -core**3 + core * ranks[5]
        
        pred = primary.ewm(span=self.span, min_periods=1).mean()
        
        # CAUSAL TILT REMOVAL (from demo_engineered.py)
        expanding_mean = pred.shift(1).expanding(min_periods=1).mean()
        pred = (pred - expanding_mean).fillna(0.0)
        
        pred = pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)
        return pred

class PrimaryWithTilt(Predictor):
    """Primary WITHOUT tilt removal (our old approach)."""
    def __init__(self):
        self.span = 72

    def train(self, features, target):
        pass

    def predict(self, features):
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        ranks = [csrank(g[i]) for i in range(1, 7)]
        core = sum(ranks[:4]) / 4.0
        primary = -core**3 + core * ranks[5]
        pred = primary.ewm(span=self.span, min_periods=1).mean()
        pred = pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)
        return pred

periods = sorted(list_periods("data/train"))

print("COMPARING: WITH vs WITHOUT causal tilt removal")
print("=" * 100)
print(f"{'Period':>6} {'NoTilt_Sh':>10} {'WithTilt_Sh':>12} {'NoTilt_IC':>10} {'WithTilt_IC':>12}")
print("=" * 100)

notilt_sh = []
withtilt_sh = []
notilt_ic = []
withtilt_ic = []

for p in periods:
    d = load_period(p, "data/train", "small")
    n = len(d["features"])
    n_val = compute_val_rows(n)
    n_train = n - n_val
    
    val_feat = d["features"].iloc[n_train:]
    val_ret = d["returns"].iloc[n_train:]
    val_tgt = d["target"].iloc[n_train:]
    
    # No tilt removal
    p1 = PrimaryNoTilt()
    pred1 = p1.predict(val_feat)
    pf1 = backtest(pred1, val_ret)
    sh1 = float(pf1.mean() / pf1.std()) if pf1.std() > 0 else 0.0
    ic1 = np.nanmean(val_tgt.corrwith(pred1, axis=1))
    
    # With tilt (no removal - our old approach)
    p2 = PrimaryWithTilt()
    pred2 = p2.predict(val_feat)
    pf2 = backtest(pred2, val_ret)
    sh2 = float(pf2.mean() / pf2.std()) if pf2.std() > 0 else 0.0
    ic2 = np.nanmean(val_tgt.corrwith(pred2, axis=1))
    
    notilt_sh.append(sh1)
    withtilt_sh.append(sh2)
    notilt_ic.append(ic1)
    withtilt_ic.append(ic2)
    
    print(f"{p:>6} {sh1:>10.4f} {sh2:>12.4f} {ic1:>10.4f} {ic2:>12.4f}")

print("=" * 100)
print(f"{'MEAN':>6} {np.mean(notilt_sh):>10.4f} {np.mean(withtilt_sh):>12.4f} {np.mean(notilt_ic):>10.4f} {np.mean(withtilt_ic):>12.4f}")
print(f"{'STD':>6} {np.std(notilt_sh):>10.4f} {np.std(withtilt_sh):>12.4f}")
print(f"{'MIN':>6} {np.min(notilt_sh):>10.4f} {np.min(withtilt_sh):>12.4f}")
print(f"NoTilt  neg periods: {sum(1 for s in notilt_sh if s < 0)}/{len(notilt_sh)}")
print(f"WithTilt neg periods: {sum(1 for s in withtilt_sh if s < 0)}/{len(withtilt_sh)}")

# Also test full walkforward
print("\n\nFULL WALKFORWARD:")
res1 = walkforward_full(PrimaryNoTilt, gauge_fix=False, verbose=False)
res2 = walkforward_full(PrimaryWithTilt, gauge_fix=False, verbose=False)
print(f"PrimaryNoTilt  Full Sharpe: {res1['sharpe']:.6f}")
print(f"PrimaryWithTilt Full Sharpe: {res2['sharpe']:.6f}")

# Test different spans with tilt removal
print("\n\nSPAN SWEEP WITH TILT REMOVAL:")
for span in [24, 48, 72, 120, 240, 720]:
    class T(Predictor):
        def __init__(self):
            self.span = span
        def train(self, f, t):
            pass
        def predict(self, features):
            g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
            ranks = [csrank(g[i]) for i in range(1, 7)]
            core = sum(ranks[:4]) / 4.0
            primary = -core**3 + core * ranks[5]
            pred = primary.ewm(span=self.span, min_periods=1).mean()
            em = pred.shift(1).expanding(min_periods=1).mean()
            pred = (pred - em).fillna(0.0)
            return pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)
    
    res = walkforward_full(T, gauge_fix=False, verbose=False)
    print(f"  Span {span:>4}: Full Sharpe = {res['sharpe']:.6f}")
