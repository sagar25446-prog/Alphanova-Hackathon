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
    "F1": -np_csrank(gi[1]),
    "-core": -np_csrank((gi[1] + gi[2] + gi[3] + gi[4]) / 4.0),
    "-core^3": -(np_csrank((gi[1] + gi[2] + gi[3] + gi[4]) / 4.0)**3),
    "primary": -(np_csrank((gi[1] + gi[2] + gi[3] + gi[4]) / 4.0)**3) + np_csrank((gi[1] + gi[2] + gi[3] + gi[4]) / 4.0) * np_csrank(gi[6]),
}

for name, signal in signals.items():
    print(f"\nEvaluating {name}")
    for span in [120, 480, 720]:
        alpha = 2.0 / (span + 1.0)
        state = np.zeros(20)
        smoothed = np.empty_like(signal)
        for row in range(len(signal)):
            state = alpha * signal[row] + (1 - alpha) * state
            smoothed[row] = state
        smoothed -= smoothed.mean(axis=1, keepdims=True)
        
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
        
        pred_oos = pred[n:]
        ret_oos = ri_arr[n:]
        
        pnl = []
        for row in range(1, len(pred_oos)):
            gross = np.nansum(pred_oos[row-1] * ret_oos[row])
            cost = 0.0005 * np.nansum(np.abs(pred_oos[row-1] - (pred_oos[row-2] if row >= 2 else np.zeros(20))))
            pnl.append(gross - cost)
        
        pnl = np.array(pnl)
        sharpe = np.mean(pnl) / np.std(pnl) if np.std(pnl) > 0 else 0
        print(f"  span={span:3d}: Sharpe={sharpe:.6f}")
