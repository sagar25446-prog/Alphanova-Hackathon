"""Deep analysis of what features predict the target cross-sectionally.

We test EVERY possible contemporaneous feature combination to find what works.
The gate requires contemporaneous features only (no rolling windows).
"""
import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")

import numpy as np
import pandas as pd
from data_loader import list_periods, load_period

DATA_DIR = "data/train"
UNIVERSE = "small"

def csrank(df):
    r = df.rank(axis=1)
    n = df.notna().sum(axis=1).values.reshape(-1, 1)
    return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

def np_csrank(arr):
    """Cross-sectional rank of numpy array"""
    order = np.argsort(np.argsort(arr, axis=1, kind="mergesort"), axis=1, kind="mergesort")
    return (order.astype(float) - 9.5) / 9.5

# Load the LATEST period for analysis
periods = list_periods(DATA_DIR, UNIVERSE)
print(f"Available periods: {periods}")

# Use last period
data = load_period(periods[-1], DATA_DIR, UNIVERSE)
features = data['features']
target = data['target']
returns = data['returns']

tickers = sorted(features["Feature.1"].columns, key=lambda name: int(name.split(".")[-1]))
print(f"Tickers: {tickers}")
print(f"Total rows: {len(features.index)}")

# Get raw features
g = {i: features[f"Feature.{i}"][tickers].values for i in range(1, 7)}
y = target[tickers].values
ret = returns[tickers].values

# Split: use last 800 rows as test
n_train = len(y) - 800
y_test = y[n_train:]
ret_test = ret[n_train:]

print(f"\nTrain rows: {n_train}, Test rows: {len(y_test)}")

# === STEP 1: Test every raw feature's cross-sectional IC with target ===
print("\n" + "="*60)
print("STEP 1: Raw feature IC with target (OOS, period 031)")
print("="*60)

for i in range(1, 7):
    # Compute rank
    r = np_csrank(g[i][n_train:])
    # Cross-sectional correlation per row
    ics = []
    for row in range(len(r)):
        if np.isfinite(y_test[row]).sum() > 5:
            ok = np.isfinite(y_test[row])
            ic = np.corrcoef(r[row][ok], y_test[row][ok])[0, 1]
            if np.isfinite(ic):
                ics.append(ic)
    print(f"  F{i} rank IC: mean={np.mean(ics):.6f}, std={np.std(ics):.6f}, t={np.mean(ics)/np.std(ics)*np.sqrt(len(ics)):.2f}")

# === STEP 2: Test feature differences ===
print("\n" + "="*60)
print("STEP 2: Feature difference IC with target")
print("="*60)

for i in range(1, 7):
    for j in range(i+1, 7):
        r = np_csrank(g[i][n_train:] - g[j][n_train:])
        ics = []
        for row in range(len(r)):
            if np.isfinite(y_test[row]).sum() > 5:
                ok = np.isfinite(y_test[row])
                ic = np.corrcoef(r[row][ok], y_test[row][ok])[0, 1]
                if np.isfinite(ic):
                    ics.append(ic)
        mean_ic = np.mean(ics)
        if abs(mean_ic) > 0.01:
            print(f"  F{i}-F{j} rank IC: mean={mean_ic:.6f}, std={np.std(ics):.6f}, t={mean_ic/np.std(ics)*np.sqrt(len(ics)):.2f}")

# === STEP 3: Test nonlinear combinations ===
print("\n" + "="*60)
print("STEP 3: Nonlinear feature IC with target")
print("="*60)

# Core rank (F1-F4 average)
core_raw = (g[1][n_train:] + g[2][n_train:] + g[3][n_train:] + g[4][n_train:]) / 4.0
core = np_csrank(core_raw)

# Primary formula: -core^3 + core * rank6
ranks_test = [np_csrank(g[i][n_train:]) for i in range(1, 7)]
primary = -core**3 + core * ranks_test[5]
sleeve = -ranks_test[4]

for name, signal in [("core", core), ("primary (-core^3+core*r6)", primary), 
                      ("sleeve (-r5)", sleeve),
                      ("primary+2*sleeve", primary + 2*sleeve),
                      ("primary+3*sleeve", primary + 3*sleeve),
                      ("core*r5", core * ranks_test[4]),
                      ("core*r6", core * ranks_test[5]),
                      ("r5*r6", ranks_test[4] * ranks_test[5]),
                      ("core^2", core**2),
                      ("-core^3", -core**3),
                      ("r1*r2", ranks_test[0] * ranks_test[1]),
                      ("r1-r5", ranks_test[0] - ranks_test[4]),
                      ("r6-r5", ranks_test[5] - ranks_test[4]),
                      ]:
    ics = []
    for row in range(len(signal)):
        if np.isfinite(y_test[row]).sum() > 5:
            ok = np.isfinite(y_test[row])
            ic = np.corrcoef(signal[row][ok], y_test[row][ok])[0, 1]
            if np.isfinite(ic):
                ics.append(ic)
    print(f"  {name:30s} IC: mean={np.mean(ics):.6f}, std={np.std(ics):.6f}, t={np.mean(ics)/np.std(ics)*np.sqrt(len(ics)):.2f}")

# === STEP 4: Test across ALL periods (not just last one) ===
print("\n" + "="*60)
print("STEP 4: Cross-period IC stability (OOS split per period)")
print("="*60)

for period in periods[-5:]:
    data = load_period(period, DATA_DIR, UNIVERSE)
    f = data['features']
    t = data['target']
    tickers = sorted(f["Feature.1"].columns, key=lambda name: int(name.split(".")[-1]))
    
    gi = {i: f[f"Feature.{i}"][tickers].values for i in range(1, 7)}
    yi = t[tickers].values
    
    n = len(yi) - 800
    if n < 100:
        continue
    
    yi_test = yi[n:]
    ri = [np_csrank(gi[i][n:]) for i in range(1, 7)]
    
    core_i = np_csrank((gi[1][n:] + gi[2][n:] + gi[3][n:] + gi[4][n:]) / 4.0)
    primary_i = -core_i**3 + core_i * ri[5]
    sleeve_i = -ri[4]
    combined_i = primary_i + 2 * sleeve_i
    
    for name, signal in [("sleeve (-r5)", sleeve_i), ("primary+2*sleeve", combined_i)]:
        ics = []
        for row in range(len(signal)):
            if np.isfinite(yi_test[row]).sum() > 5:
                ok = np.isfinite(yi_test[row])
                ic = np.corrcoef(signal[row][ok], yi_test[row][ok])[0, 1]
                if np.isfinite(ic):
                    ics.append(ic)
        print(f"  Period {period} {name:25s} IC: mean={np.mean(ics):.6f}")

# === STEP 5: Sharpe simulation with different smoothing spans ===
print("\n" + "="*60)
print("STEP 5: Sharpe with different smoothing (period 031 OOS)")
print("="*60)

data = load_period(periods[-1], DATA_DIR, UNIVERSE)
f = data['features']
t = data['target']
ret = data['returns']
tickers = sorted(f["Feature.1"].columns, key=lambda name: int(name.split(".")[-1]))

gi = {i: f[f"Feature.{i}"][tickers].values for i in range(1, 7)}
yi = t[tickers].values
ri_arr = ret[tickers].values

n = len(yi) - 800
ri_all = [np_csrank(gi[i]) for i in range(1, 7)]
core_all = np_csrank((gi[1] + gi[2] + gi[3] + gi[4]) / 4.0)
primary_all = -core_all**3 + core_all * ri_all[5]
sleeve_all = -ri_all[4]
combined_all = primary_all + 2.0 * sleeve_all

for span in [60, 120, 240, 480, 720, 1440]:
    # EWM smoothing
    alpha = 2.0 / (span + 1.0)
    state = np.zeros(20)
    smoothed = np.empty_like(combined_all)
    for row in range(len(combined_all)):
        state = alpha * combined_all[row] + (1 - alpha) * state
        smoothed[row] = state
    smoothed -= smoothed.mean(axis=1, keepdims=True)
    
    # Causal tilt removal
    expanding_sum = np.zeros(20)
    expanding_count = 0
    pred = np.empty_like(smoothed)
    for row in range(len(smoothed)):
        if row == 0:
            pred[row] = smoothed[row]
        else:
            expanding_sum += smoothed[row-1]
            expanding_count += 1
            mean = expanding_sum / expanding_count
            pred[row] = smoothed[row] - mean
        pred[row] -= pred[row].mean()
    
    # Sharpe on OOS
    pred_oos = pred[n:]
    ret_oos = ri_arr[n:]
    
    pnl = []
    for row in range(1, len(pred_oos)):
        gross = np.nansum(pred_oos[row-1] * ret_oos[row])
        cost = 0.0005 * np.nansum(np.abs(pred_oos[row-1] - (pred_oos[row-2] if row >= 2 else np.zeros(20))))
        pnl.append(gross - cost)
    
    pnl = np.array(pnl)
    sharpe = np.mean(pnl) / np.std(pnl) if np.std(pnl) > 0 else 0
    print(f"  span={span:5d}: Sharpe={sharpe:.6f}, mean_pnl={np.mean(pnl):.8f}, std_pnl={np.std(pnl):.8f}")
