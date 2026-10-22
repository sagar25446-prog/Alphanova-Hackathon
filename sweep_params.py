"""Sweep key parameters on inverted_f5blend to find optimal server config.

The inverted_f5blend scored -0.0068 on server (rank 184). 
Key insight: server gap from local is ~+0.037.
So to get server ~0.0, need local ~-0.037.
To get server +0.005, need local ~-0.042.
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from walkforward import walkforward_full

DATA_DIR = "data/train"
UNIVERSE = "small"

# Import the base class to modify
from alphanova_inverted_f5blend import InvertedPurgedPrismF5Blend

configs = [
    # (primary_span, sleeve_span, weight, label)
    (120, 480, 2.0, "span120_s480_w2"),   # Less smooth, more signal
    (120, 600, 2.0, "span120_s600_w2"),   # Less smooth primary, slow sleeve
    (120, 600, 3.0, "span120_s600_w3"),   # Heavy sleeve weight
    (140, 600, 2.0, "span140_s600_w2"),   # Slightly more smooth
    (160, 600, 2.0, "span160_s600_w2"),   # CURRENT BEST (baseline)
    (160, 600, 3.0, "span160_s600_w3"),   # More sleeve weight  
    (160, 720, 2.0, "span160_s720_w2"),   # Slower sleeve
    (160, 720, 3.0, "span160_s720_w3"),   # Slower sleeve + more weight
    (200, 600, 2.0, "span200_s600_w2"),   # More smooth
    (200, 720, 3.0, "span200_s720_w3"),   # More of everything
    (120, 600, 4.0, "span120_s600_w4"),   # Very heavy sleeve
]

print(f"{'Config':<25s} {'Local Sharpe':>14s}  {'Predicted Server':>16s}")
print("-" * 60)

for pspan, sspan, w, label in configs:
    class Variant(InvertedPurgedPrismF5Blend):
        def __init__(self):
            super().__init__()
            self.primary_span = pspan
            self.sleeve_span = sspan
            self.weight = w
    
    try:
        results = walkforward_full(Variant, DATA_DIR, UNIVERSE, verbose=False)
        sharpe = results.get("sharpe", 0)
        predicted_server = sharpe + 0.037  # estimated gap
        print(f"{label:<25s} {sharpe:>+14.6f}  {predicted_server:>+16.6f}")
    except Exception as e:
        print(f"{label:<25s} ERROR: {e}")
