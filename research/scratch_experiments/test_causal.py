import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
from alphanova_causal_prism import CausalPrismPredictor
from walkforward import walkforward_full

if __name__ == "__main__":
    res = walkforward_full(CausalPrismPredictor, gauge_fix=False, verbose=False)
    print(f"Causal Prism Full Sharpe: {res['sharpe']:.6f}")
