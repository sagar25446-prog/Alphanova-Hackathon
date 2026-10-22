import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
sys.path.insert(0, r"C:\Users\Sagar Jain\.gemini\antigravity\brain\23c3aefc-23d4-4c83-bcf6-595cd4899e13\scratch")
from test_smooth_features import SmoothFeaturesPredictor
from walkforward import walkforward_full

res = walkforward_full(SmoothFeaturesPredictor, gauge_fix=False, verbose=False)
print(f"Sharpe: {res['sharpe']:.4f}")
print(f"IC: {res['ic']:.4f}")
