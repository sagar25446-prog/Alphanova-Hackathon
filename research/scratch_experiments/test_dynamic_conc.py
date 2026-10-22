import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
import numpy as np
from scratch.test_dynamic_power import TopRankPredictor
from data_loader import load_period

d = load_period('031', 'data/train', 'small')
feat = d['features']

p = TopRankPredictor()
pred = p.predict(feat)

# Normalize L1 sum to 1 per row (since portfolio assumes weights sum to 1 in abs)
pred_norm = pred.div(pred.abs().sum(axis=1), axis=0).fillna(0)

print("Concentration (L2 norm squared of weights):", (pred_norm**2).sum(axis=1).mean())
print("Max absolute weight:", pred_norm.abs().max(axis=1).mean())
