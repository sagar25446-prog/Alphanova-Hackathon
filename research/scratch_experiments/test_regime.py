import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
import numpy as np
import pandas as pd
from alphanova_causal_prism import CausalPrismPredictor
from walkforward import walkforward_full

for strength in [0.0, 0.01, 0.05, 0.10, 0.20, 0.50]:
    class T(CausalPrismPredictor):
        def __init__(self):
            super().__init__()
            self.regime_strength = strength
            self.span = 130
    
    res = walkforward_full(T, gauge_fix=False, verbose=False)
    print(f"regime_strength={strength:.2f}, span=130: Full Sharpe = {res['sharpe']:.6f}")
