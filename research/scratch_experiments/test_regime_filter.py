"""Test if we can use the train period to filter/invert the signal."""
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

train_sharpes = []
val_sharpes = []

for p in periods:
    d = load_period(p, "data/train", "small")
    n = len(d["features"])
    n_val = compute_val_rows(n)
    n_train = n - n_val
    
    f = d["features"]
    r = d["returns"]
    
    g = {i: csrank(f[f"Feature.{i}"]) for i in range(1, 7)}
    core = sum(g[i] for i in range(1, 5)) / 4.0
    sig = -core**3 + core * g[6]
    
    pred = sig.ewm(span=72, min_periods=1).mean()
    pred = pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)
    
    pred_train = pred.iloc[:n_train]
    ret_train = r.iloc[:n_train]
    pf_train = backtest(pred_train, ret_train)
    sh_train = float(pf_train.mean() / pf_train.std()) if pf_train.std() > 0 else 0.0
    
    pred_val = pred.iloc[n_train:]
    ret_val = r.iloc[n_train:]
    pf_val = backtest(pred_val, ret_val)
    sh_val = float(pf_val.mean() / pf_val.std()) if pf_val.std() > 0 else 0.0
    
    train_sharpes.append(sh_train)
    val_sharpes.append(sh_val)
    
    print(f"{p:>6} TrainSh: {sh_train:>8.4f}  ValSh: {sh_val:>8.4f}")

train_sharpes = np.array(train_sharpes)
val_sharpes = np.array(val_sharpes)

print("=" * 40)
print(f"Correlation Train vs Val: {np.corrcoef(train_sharpes, val_sharpes)[0,1]:.4f}")

# Simulating a dynamic predictor:
# If train_sharpe < 0, invert the signal.
dyn_sharpes = np.where(train_sharpes < 0, -val_sharpes, val_sharpes)
print(f"Static Mean Sharpe: {np.mean(val_sharpes):.4f}")
print(f"Dynamic Mean Sharpe: {np.mean(dyn_sharpes):.4f}")
print(f"Static Neg Pct: {np.mean(val_sharpes < 0)*100:.1f}%")
print(f"Dynamic Neg Pct: {np.mean(dyn_sharpes < 0)*100:.1f}%")
