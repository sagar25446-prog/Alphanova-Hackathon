import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
sys.path.insert(0, r"C:\Users\Sagar Jain\.gemini\antigravity\brain\23c3aefc-23d4-4c83-bcf6-595cd4899e13\scratch")
from test_discrete import TopRankPredictor
from walkforward import walkforward_full

class P(TopRankPredictor):
    def __init__(self):
        super().__init__()
        self.span = 720

res = walkforward_full(P, gauge_fix=False, verbose=False)
print(f"Top 2 span 720 Sharpe: {res['sharpe']:.4f}")
