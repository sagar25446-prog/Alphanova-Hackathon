import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
from alphanova_rank1_ultimate import SmoothedPrimaryPredictor
from walkforward import walkforward_full

res = walkforward_full(SmoothedPrimaryPredictor, gauge_fix=False, verbose=False)
print(f"Ultimate Full Sharpe: {res['sharpe']:.6f}")
