import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
from scratch.dynamic_formula import TopRankPredictor
from walkforward import walkforward_full

print("Testing Dynamic Formula:")
res = walkforward_full(TopRankPredictor, gauge_fix=False, verbose=False)
print(f"Overall Sharpe: {res['sharpe']:.4f}")
