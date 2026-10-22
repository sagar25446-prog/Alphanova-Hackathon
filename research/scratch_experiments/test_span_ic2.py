import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
import numpy as np
import pandas as pd
from test_span import SpanPredictor
from walkforward import walkforward_backtest

class T72(SpanPredictor):
    def __init__(self):
        super().__init__(span=72)

class T80(SpanPredictor):
    def __init__(self):
        super().__init__(span=80)

for cls, s in [(T72, 72), (T80, 80)]:
    res = walkforward_backtest(cls, verbose=False)
    pred = res['predictions']
    t = pd.read_parquet('data/train/031.small.target.parquet').reindex(pred.index)
    ic = np.nanmean(t.corrwith(pred, axis=1))
    print(f"Span {s} IC Target: {ic:.4f}")
