import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
import numpy as np
import pandas as pd
from data_loader import load_period

def csrank(df):
    r = df.rank(axis=1)
    n = df.notna().sum(axis=1).values.reshape(-1, 1)
    return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

d = load_period('031', 'data/train', 'small')
f = d['features']

g = {i: csrank(f[f"Feature.{i}"]) for i in range(1, 7)}

for i in range(1, 7):
    for j in range(i+1, 7):
        # Cross-sectional correlation is just the mean of the row-wise dot products (since they are demeaned and scaled)
        # Or easier: just flatten and correlate
        corr = np.corrcoef(g[i].values.flatten()[~np.isnan(g[i].values.flatten())], 
                           g[j].values.flatten()[~np.isnan(g[j].values.flatten())])[0,1]
        print(f"F{i} vs F{j}: {corr:.4f}")

print("-" * 30)
# Let's also check Target IC
t = d['target']
for i in range(1, 7):
    ic = np.nanmean(t.corrwith(g[i], axis=1))
    print(f"F{i} Target IC: {ic:.4f}")

