import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
import numpy as np
import pandas as pd
from test_span import SpanPredictor
from walkforward import walkforward_backtest

class T120(SpanPredictor):
    def __init__(self):
        super().__init__(span=120)

class T48(SpanPredictor):
    def __init__(self):
        super().__init__(span=48)

for cls, s in [(T120, 120), (T48, 48)]:
    res = walkforward_backtest(cls, verbose=False)
    pred = res['predictions']
    t = pd.read_parquet('data/train/031.small.target.parquet').reindex(pred.index)
    ic = np.nanmean(t.corrwith(pred, axis=1))
    print(f"Span {s} IC Target: {ic:.4f}")
