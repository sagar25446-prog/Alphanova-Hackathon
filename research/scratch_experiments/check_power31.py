import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
sys.path.insert(0, r"C:\Users\Sagar Jain\.gemini\antigravity\brain\23c3aefc-23d4-4c83-bcf6-595cd4899e13\scratch")
from test_powers_smooth_features import PowerSmoothFeaturesPredictor
from walkforward import walkforward_full

for p_val in [21, 31, 51]:
    class P(PowerSmoothFeaturesPredictor):
        def __init__(self):
            super().__init__(power=p_val)
    res = walkforward_full(P, gauge_fix=False, verbose=False)
    print(f"Power {p_val:>2} (Smooth Features): Sharpe = {res['sharpe']:.4f}")
