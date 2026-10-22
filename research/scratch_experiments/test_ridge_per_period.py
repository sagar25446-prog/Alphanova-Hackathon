"""Test if Ridge on Target works in ANY period."""
import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from data_loader import list_periods, load_period
from evaluation import backtest
from walkforward import compute_val_rows

def csrank(df):
    r = df.rank(axis=1)
    n = df.notna().sum(axis=1).values.reshape(-1, 1)
    return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

periods = sorted(list_periods("data/train"))

print(f"{'Period':>6} {'RidgeSh':>10} {'RidgeIC':>10}")
print("=" * 30)

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
    
    # Simple cross-sectional rank of all 6 features
    g_train = {i: csrank(train_feat[f"Feature.{i}"]) for i in range(1, 7)}
    g_val = {i: csrank(val_feat[f"Feature.{i}"]) for i in range(1, 7)}
    
    X_train = np.column_stack([g_train[i].values.reshape(-1) for i in range(1, 7)])
    y_train = train_tgt.values.reshape(-1)
    
    X_val = np.column_stack([g_val[i].values.reshape(-1) for i in range(1, 7)])
    
    # Train Ridge
    ok = np.isfinite(y_train) & np.isfinite(X_train).all(axis=1)
    model = Ridge(alpha=100.0)
    model.fit(X_train[ok], y_train[ok])
    
    # Predict
    pred_val = model.predict(np.nan_to_num(X_val))
    pred_val = pd.DataFrame(pred_val.reshape(-1, val_feat.shape[1] // 6), index=val_feat.index)
    
    # Smooth
    pred_val = pred_val.ewm(span=72, min_periods=1).mean()
    
    # Demean
    pred_val = pred_val.sub(pred_val.mean(axis=1), axis=0).fillna(0.0)
    
    # Evaluate
    pf = backtest(pred_val, val_ret)
    sh = float(pf.mean() / pf.std()) if pf.std() > 0 else 0.0
    ic = np.nanmean(val_tgt.corrwith(pred_val, axis=1))
    
    sharpes.append(sh)
    ics.append(ic)
    
    print(f"{p:>6} {sh:>10.4f} {ic:>10.4f}")

print("=" * 30)
print(f"MEAN   {np.mean(sharpes):>10.4f} {np.mean(ics):>10.4f}")
