import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
import numpy as np
import pandas as pd
from test_span import SpanPredictor
from walkforward import walkforward_backtest, walkforward_full

class T53(SpanPredictor):
    def __init__(self):
        super().__init__(span=53)

res = walkforward_backtest(T53, verbose=False)
pred = res['predictions']
t = pd.read_parquet('data/train/031.small.target.parquet').reindex(pred.index)
ic = np.nanmean(t.corrwith(pred, axis=1))
print(f"Span 53 IC Target: {ic:.4f}")

full = walkforward_full(T53, gauge_fix=False, verbose=False)
print(f"Span 53 Full Sharpe: {full['sharpe']:.6f}")
