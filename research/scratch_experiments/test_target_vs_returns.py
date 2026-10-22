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

core = np_csrank((gi[1] + gi[2] + gi[3] + gi[4]) / 4.0)
primary = -core**3 + core * np_csrank(gi[6])

# IN-SAMPLE (with TARGET)
print("IN-SAMPLE with TARGET:")
ics_train_t = []
for row in range(n):
    if np.isfinite(yi[row]).sum() > 5:
        ok = np.isfinite(yi[row])
        ic = np.corrcoef(primary[row][ok], yi[row][ok])[0, 1]
        if np.isfinite(ic):
            ics_train_t.append(ic)
print(f"  Primary IC with TARGET (in-sample): mean={np.mean(ics_train_t):.6f}")

# IN-SAMPLE (with RETURNS)
print("\nIN-SAMPLE with RETURNS:")
ics_train_r = []
for row in range(n):
    if np.isfinite(ri_arr[row]).sum() > 5:
        ok = np.isfinite(ri_arr[row])
        ic = np.corrcoef(primary[row][ok], ri_arr[row][ok])[0, 1]
        if np.isfinite(ic):
            ics_train_r.append(ic)
print(f"  Primary IC with RETURNS (in-sample): mean={np.mean(ics_train_r):.6f}")

# OUT-OF-SAMPLE (with TARGET)
print("\nOUT-OF-SAMPLE with TARGET:")
ics_oos_t = []
for row in range(n, len(yi)):
    if np.isfinite(yi[row]).sum() > 5:
        ok = np.isfinite(yi[row])
        ic = np.corrcoef(primary[row][ok], yi[row][ok])[0, 1]
        if np.isfinite(ic):
            ics_oos_t.append(ic)
print(f"  Primary IC with TARGET (OOS): mean={np.mean(ics_oos_t):.6f}")

# OUT-OF-SAMPLE (with RETURNS)
print("\nOUT-OF-SAMPLE with RETURNS:")
ics_oos_r = []
for row in range(n, len(yi)):
    if np.isfinite(ri_arr[row]).sum() > 5:
        ok = np.isfinite(ri_arr[row])
        ic = np.corrcoef(primary[row][ok], ri_arr[row][ok])[0, 1]
        if np.isfinite(ic):
            ics_oos_r.append(ic)
print(f"  Primary IC with RETURNS (OOS): mean={np.mean(ics_oos_r):.6f}")
