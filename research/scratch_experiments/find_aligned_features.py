"""Analyze all features to find ALIGNED features vs TRAP features."""
import pandas as pd
import numpy as np
from data_loader import list_periods, load_period

def compute_ics(period="031"):
    d = load_period(period, "data/train", "small")
    f = {i: d["features"][f"Feature.{i}"] for i in range(1, 7)}
    t = d["target"]
    r = d["returns"]
    
    def get_ranks(df):
        ranks = df.rank(axis=1)
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (ranks - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

    ranks = {i: get_ranks(f[i]) for i in range(1, 7)}
    
    # Generate candidate features
    candidates = {}
    for i in range(1, 7):
        candidates[f"F{i}"] = ranks[i]
        
    for i in range(1, 7):
        for j in range(i+1, 7):
            candidates[f"F{i}-F{j}"] = get_ranks(ranks[i] - ranks[j])
            candidates[f"F{i}+F{j}"] = get_ranks(ranks[i] + ranks[j])
            candidates[f"F{i}*F{j}"] = get_ranks(ranks[i] * ranks[j])
            
    core = sum(ranks[i] for i in range(1, 5)) / 4.0
    candidates["core"] = core
    candidates["primary"] = -core**3 + core * ranks[6]
    candidates["negcore3"] = -core**3
    
    results = []
    for name, feat in candidates.items():
        ic_target = np.nanmean(t.corrwith(feat, axis=1))
        ic_return = np.nanmean(r.corrwith(feat, axis=1))
        # If sign is same, aligned. If opposite, trap.
        aligned = np.sign(ic_target) == np.sign(ic_return)
        
        # We want to know what happens if we train on Target.
        # If we train on Target, the model learns a weight with the same sign as ic_target.
        # The resulting prediction has correlation with Return equal to:
        # sign(ic_target) * ic_return
        # So expected PnL direction is sign(ic_target) * ic_return.
        # Wait, if IC_target is +, weight is +. Pred = +feat. IC_Return of pred is +ic_return.
        # If IC_target is -, weight is -. Pred = -feat. IC_Return of pred is -ic_return.
        # So Expected PnL is proportional to IC_target * IC_return !
        expected_pnl = ic_target * ic_return
        
        results.append({
            "feature": name,
            "ic_target": ic_target,
            "ic_return": ic_return,
            "expected_pnl": expected_pnl,
            "aligned": aligned
        })
        
    df = pd.DataFrame(results).sort_values("expected_pnl", ascending=False)
    return df

df = compute_ics("031")
print("PERIOD 031 - ALIGNMENT ANALYSIS")
print(df.to_string(index=False))

# Let's check another period to see if it's stable
df20 = compute_ics("020")
print("\nPERIOD 020 - ALIGNMENT ANALYSIS")
print(df20.to_string(index=False))
