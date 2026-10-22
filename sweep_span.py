"""Sweep smoothing spans on the inverted minimal prism to find optimal turnover."""
import sys, time
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from data_loader import list_periods, load_period
from walkforward import walkforward_full
from alphanova_inverted_minimal import InvertedMinimalPrism

DATA_DIR = "data/train"
UNIVERSE = "small"

for span in [120, 180, 240, 360, 480, 720]:
    class TestPredictor(InvertedMinimalPrism):
        def __init__(self):
            super().__init__()
            self.span = span
    
    try:
        results = walkforward_full(TestPredictor, DATA_DIR, UNIVERSE, verbose=False)
        sharpe = results.get("sharpe", 0)
        print(f"Span={span:4d}  Sharpe={sharpe:+.6f}")
    except Exception as e:
        print(f"Span={span:4d}  ERROR: {e}")
