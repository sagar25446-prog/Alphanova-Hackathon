import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
import numpy as np
import pandas as pd
from predictor import Predictor
from walkforward import walkforward_full

class SpanPredictor(Predictor):
    def __init__(self, span=24):
        self.span = span

    def train(self, features, target):
        pass

    def predict(self, features):
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        
        def csrank(df):
            r = df.rank(axis=1)
            n = df.notna().sum(axis=1).values.reshape(-1, 1)
            return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

        ranks = [csrank(g[i]) for i in range(1, 7)]
        core = sum(ranks[:4]) / 4.0
        primary = -core**3 + core * ranks[5]
        
        pred = primary.ewm(span=self.span, min_periods=1).mean()
        pred = pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)
        return pred

if __name__ == "__main__":
    spans = [2, 4, 8, 12, 24, 48, 72, 120, 240, 360, 720]
    results = []
    
    for s in spans:
        print(f"Testing span={s}...")
        
        # We need a Predictor class that takes no args in __init__ for walkforward_full
        class TempPredictor(SpanPredictor):
            def __init__(self):
                super().__init__(span=s)
                
        res = walkforward_full(TempPredictor, gauge_fix=False, verbose=False)
        
        # We also need Avg IC. Let's approximate by picking last period.
        # But walkforward_full doesn't return predictions for all periods easily.
        # Let's just trust the Sharpe for now, we want Sharpe > 0.017!
        print(f"Span {s}: Full Sharpe = {res['sharpe']:.6f}")
