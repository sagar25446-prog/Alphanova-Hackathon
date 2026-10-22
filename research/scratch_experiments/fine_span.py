import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
import numpy as np
import pandas as pd
from predictor import Predictor
from walkforward import walkforward_full

class SpanPredictor(Predictor):
    def __init__(self, span=120):
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
    spans = [80, 90, 100, 110, 120, 130, 150, 180]
    for s in spans:
        class T(SpanPredictor):
            def __init__(self):
                super().__init__(span=s)
        res = walkforward_full(T, gauge_fix=False, verbose=False)
        print(f"Span {s:4d}: Full Sharpe = {res['sharpe']:.6f}")
