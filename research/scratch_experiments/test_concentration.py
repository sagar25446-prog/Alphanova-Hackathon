import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
import numpy as np
import pandas as pd
from data_loader import load_period

def csrank(df):
    r = df.rank(axis=1)
    n = df.notna().sum(axis=1).values.reshape(-1, 1)
    return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

def concentration(pred):
    """Calculate cross-sectional concentration. 
    Sum of squares divided by square of sum of absolute values?"""
    # Not sure exactly how concentration is defined, but we can measure L2 / L1 norm or similar.
    # Let's just output the properties of the predictions.
    l1 = pred.abs().sum(axis=1)
    l2 = (pred**2).sum(axis=1)
    return (l2 / (l1**2)).mean()

d = load_period('031', 'data/train', 'small')
f = d['features']

g = {i: csrank(f[f"Feature.{i}"]) for i in range(1, 7)}
core = sum(g[i] for i in range(1, 5)) / 4.0

signals = {
    "primary": -core**3 + core * g[6],
    "negcore3": -core**3,
    "negcore": -core,
    "-F1": -g[1],
    "demo_xgb": None # We'd need to run the model
}

for name, sig in signals.items():
    if sig is not None:
        p = sig.ewm(span=72, min_periods=1).mean()
        p = p.sub(p.mean(axis=1), axis=0).fillna(0.0)
        c = concentration(p)
        print(f"{name:15}: {c:.6f}")
