import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
import numpy as np
import pandas as pd
from data_loader import list_periods, load_period

DATA_DIR = "data/train"
UNIVERSE = "small"
periods = list_periods(DATA_DIR, UNIVERSE)

def np_csrank(arr):
    order = np.argsort(np.argsort(arr, axis=1, kind="mergesort"), axis=1, kind="mergesort")
    return (order.astype(float) - 9.5) / 9.5

data = load_period(periods[-1], DATA_DIR, UNIVERSE)
f = data['features']
t = data['target']
ret = data['returns']
tickers = sorted(f["Feature.1"].columns, key=lambda name: int(name.split(".")[-1]))

gi = {i: f[f"Feature.{i}"][tickers].values for i in range(1, 7)}
yi = t[tickers].values
ri_arr = ret[tickers].values

n = len(yi) - 800

signals = {
    "F1": gi[1],
    "-F1": -gi[1],
    "F5": gi[5],
    "-F5 (sleeve)": -gi[5],
    "F6": gi[6],
    "core": (gi[1] + gi[2] + gi[3] + gi[4]) / 4.0,
    "primary": -(np_csrank((gi[1] + gi[2] + gi[3] + gi[4]) / 4.0)**3) + np_csrank((gi[1] + gi[2] + gi[3] + gi[4]) / 4.0) * np_csrank(gi[6]),
}

print("IN-SAMPLE CORRELATIONS (50,000 rows):")
print(f"{'Signal':<20} | {'IC with TARGET':<15} | {'IC with RETURNS':<15}")
print("-" * 55)

for name, sig_raw in signals.items():
    sig = np_csrank(sig_raw)
    
    ic_t = []
    ic_r = []
    for row in range(n):
        if np.isfinite(yi[row]).sum() > 5:
            ok = np.isfinite(yi[row])
            ic = np.corrcoef(sig[row][ok], yi[row][ok])[0, 1]
            if np.isfinite(ic):
                ic_t.append(ic)
                
        if np.isfinite(ri_arr[row]).sum() > 5:
            ok = np.isfinite(ri_arr[row])
            ic = np.corrcoef(sig[row][ok], ri_arr[row][ok])[0, 1]
            if np.isfinite(ic):
                ic_r.append(ic)
                
    print(f"{name:<20} | {np.mean(ic_t):>15.6f} | {np.mean(ic_r):>15.6f}")

