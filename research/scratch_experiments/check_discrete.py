import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
sys.path.insert(0, r"C:\Users\Sagar Jain\.gemini\antigravity\brain\23c3aefc-23d4-4c83-bcf6-595cd4899e13\scratch")
from test_discrete import TopRankPredictor
from walkforward import walkforward_full
from data_loader import load_period
import numpy as np

class P(TopRankPredictor):
    def __init__(self):
        super().__init__()
        self.top_n = 2

res = walkforward_full(P, gauge_fix=False, verbose=False)
print(f"Sharpe: {res['sharpe']:.4f}")

p = P()
d = load_period('031', 'data/train', 'small')
pred = p.predict(d['features'])
weights = pred.div(pred.abs().sum(axis=1), axis=0).fillna(0)
conc = (weights**2).sum(axis=1).mean()
print(f"Concentration (with EWM 240): {conc:.4f}")
