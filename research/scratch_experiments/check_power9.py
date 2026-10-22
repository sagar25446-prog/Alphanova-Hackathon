import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
sys.path.insert(0, r"C:\Users\Sagar Jain\.gemini\antigravity\brain\23c3aefc-23d4-4c83-bcf6-595cd4899e13\scratch")
from test_proper_power import ProperPowerPredictor
from walkforward import walkforward_full
from data_loader import load_period

for p in [9, 10]:
    class P(ProperPowerPredictor):
        def __init__(self):
            super().__init__(power=p)
    res = walkforward_full(P, gauge_fix=False, verbose=False)
    d = load_period('031', 'data/train', 'small')
    pred = P().predict(d['features'])
    w = pred.div(pred.abs().sum(axis=1), axis=0).fillna(0)
    print(f"Power {p}: Sharpe={res['sharpe']:.4f}, Conc={(w**2).sum(axis=1).mean():.4f}")
