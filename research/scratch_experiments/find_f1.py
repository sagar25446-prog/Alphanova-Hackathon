import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
import numpy as np
import pandas as pd
from data_loader import load_period

d = load_period('031', 'data/train', 'small')
f = d['features']

def csrank(df):
    r = df.rank(axis=1)
    n = df.notna().sum(axis=1).values.reshape(-1, 1)
    return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

g = {i: csrank(f[f"Feature.{i}"]) for i in range(1, 7)}
corrs = np.zeros((6, 6))

for i in range(1, 7):
    for j in range(1, 7):
        v1 = g[i].values.flatten()
        v2 = g[j].values.flatten()
        valid = ~np.isnan(v1) & ~np.isnan(v2)
        corrs[i-1, j-1] = np.corrcoef(v1[valid], v2[valid])[0,1]

print("Cross-sectional correlation matrix:")
print(np.round(corrs, 2))

# Which feature has the highest average correlation with the others?
mean_abs_corr = np.mean(np.abs(corrs), axis=1)
print("\nMean absolute correlation per feature:")
for i in range(1, 7):
    print(f"F{i}: {mean_abs_corr[i-1]:.2f}")
    
# Variance? 
print("\nVariance per feature (should be all similar since it's rank):")
for i in range(1, 7):
    print(f"F{i}: {np.nanvar(g[i].values):.4f}")
