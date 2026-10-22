import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
import numpy as np
import pandas as pd
from test_span import SpanPredictor
from walkforward import walkforward_full

class T72(SpanPredictor):
    def __init__(self):
        super().__init__(span=72)

res = walkforward_full(T72, gauge_fix=True, verbose=False)
print(f"Span 72 Full IC: {res['ic']:.4f}")
print(f"Span 72 Full Sharpe: {res['sharpe']:.6f}")
