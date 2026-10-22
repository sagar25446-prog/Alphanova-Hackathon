"""Analyze predictive correlation of features with returns."""
import pandas as pd
import numpy as np
from data_loader import load_period

def compute_predictive_ics(period="031"):
    d = load_period(period, "data/train", "small")
    f = {i: d["features"][f"Feature.{i}"] for i in range(1, 7)}
    t = d["target"]
    r = d["returns"]
    
    def get_ranks(df):
        ranks = df.rank(axis=1)
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (ranks - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

    ranks = {i: get_ranks(f[i]) for i in range(1, 7)}
    
    candidates = {}
    for i in range(1, 7):
        candidates[f"F{i}"] = ranks[i]
        
    core = sum(ranks[i] for i in range(1, 5)) / 4.0
    candidates["core"] = core
    candidates["primary"] = -core**3 + core * ranks[6]
    candidates["negcore3"] = -core**3
    
    results = []
    for name, feat in candidates.items():
        # Predictive IC: corr(feat_{t-1}, r_t)
        # Shift feature forward by 1 so its index aligns with r_t
        feat_shifted = feat.shift(1)
        ic_return_pred = np.nanmean(r.corrwith(feat_shifted, axis=1))
        
        # Target IC: does Target correlate with feat?
        # Target is forward-looking! So target_t represents future returns.
        # Thus, corr(feat_t, target_t) is correct.
        ic_target = np.nanmean(t.corrwith(feat, axis=1))
        
        expected_pnl = ic_target * ic_return_pred
        
        results.append({
            "feature": name,
            "ic_target": ic_target,
            "ic_return_pred": ic_return_pred,
            "expected_pnl": expected_pnl,
        })
        
    df = pd.DataFrame(results).sort_values("expected_pnl", ascending=False)
    return df

print("PERIOD 031 - PREDICTIVE ALIGNMENT")
print(compute_predictive_ics("031").to_string(index=False))

print("\nPERIOD 020 - PREDICTIVE ALIGNMENT")
print(compute_predictive_ics("020").to_string(index=False))
