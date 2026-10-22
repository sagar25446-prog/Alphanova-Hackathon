import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
import numpy as np
import pandas as pd
from data_loader import load_period
from scratch.test_dynamic_power import TopRankPredictor

d = load_period('031', 'data/train', 'small')
feat = d['features']

p = TopRankPredictor()
pred = p.predict(feat)

# The evaluation script calculates concentration as the sum of squared weights, where weights sum to 1 in absolute value per timestamp.
# Let's normalize L1 to 1 per row
weights = pred.div(pred.abs().sum(axis=1), axis=0).fillna(0)
conc = (weights ** 2).sum(axis=1).mean()
print(f"Concentration of Power 3 formula: {conc:.4f}")

# What if we use power 5?
p_power5 = TopRankPredictor()
# We monkeypatch the predict method just to test power 5
def predict_5(features):
    feature_names = features.columns.get_level_values(0).unique()
    g = {f: p_power5._csrank(features[f]) for f in feature_names}
    
    n_feats = len(feature_names)
    corrs = np.zeros((n_feats, n_feats))
    f_list = list(feature_names)
    
    for i in range(n_feats):
        v1 = g[f_list[i]].values.flatten()
        valid1 = ~np.isnan(v1)
        for j in range(i, n_feats):
            v2 = g[f_list[j]].values.flatten()
            valid = valid1 & ~np.isnan(v2)
            if valid.sum() > 0:
                c = np.corrcoef(v1[valid], v2[valid])[0,1]
            else:
                c = 0.0
            corrs[i, j] = c
            corrs[j, i] = c
            
    mean_abs_corr = np.mean(np.abs(corrs), axis=1)
    sorted_indices = np.argsort(mean_abs_corr)
    
    f6_name = f_list[sorted_indices[0]]
    f5_name = f_list[sorted_indices[1]]
    core_names = [f_list[idx] for idx in sorted_indices[2:]]
    
    core = sum(g[name] for name in core_names) / 4.0
    core = p_power5._csrank(core)
    f6 = p_power5._csrank(g[f6_name])
    
    raw_pred = -(core ** 3) + core * f6
    # POWER 5
    raw_pred = np.sign(raw_pred) * (np.abs(raw_pred) ** 5)
    
    pred = raw_pred.ewm(span=120, min_periods=1).mean()
    expanding_mean = pred.shift(1).expanding(min_periods=1).mean()
    pred = (pred - expanding_mean).fillna(0.0)
    return pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)

pred5 = predict_5(feat)
weights5 = pred5.div(pred5.abs().sum(axis=1), axis=0).fillna(0)
conc5 = (weights5 ** 2).sum(axis=1).mean()
print(f"Concentration of Power 5 formula: {conc5:.4f}")

# What if we use power 9?
def predict_9(features):
    # Same as above but power 9
    feature_names = features.columns.get_level_values(0).unique()
    g = {f: p_power5._csrank(features[f]) for f in feature_names}
    
    n_feats = len(feature_names)
    corrs = np.zeros((n_feats, n_feats))
    f_list = list(feature_names)
    
    for i in range(n_feats):
        v1 = g[f_list[i]].values.flatten()
        valid1 = ~np.isnan(v1)
        for j in range(i, n_feats):
            v2 = g[f_list[j]].values.flatten()
            valid = valid1 & ~np.isnan(v2)
            if valid.sum() > 0:
                c = np.corrcoef(v1[valid], v2[valid])[0,1]
            else:
                c = 0.0
            corrs[i, j] = c
            corrs[j, i] = c
            
    mean_abs_corr = np.mean(np.abs(corrs), axis=1)
    sorted_indices = np.argsort(mean_abs_corr)
    
    f6_name = f_list[sorted_indices[0]]
    core_names = [f_list[idx] for idx in sorted_indices[2:]]
    
    core = sum(g[name] for name in core_names) / 4.0
    core = p_power5._csrank(core)
    f6 = p_power5._csrank(g[f6_name])
    
    raw_pred = -(core ** 3) + core * f6
    # POWER 9
    raw_pred = np.sign(raw_pred) * (np.abs(raw_pred) ** 9)
    
    pred = raw_pred.ewm(span=120, min_periods=1).mean()
    expanding_mean = pred.shift(1).expanding(min_periods=1).mean()
    pred = (pred - expanding_mean).fillna(0.0)
    return pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)

pred9 = predict_9(feat)
weights9 = pred9.div(pred9.abs().sum(axis=1), axis=0).fillna(0)
conc9 = (weights9 ** 2).sum(axis=1).mean()
print(f"Concentration of Power 9 formula: {conc9:.4f}")

