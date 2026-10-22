import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
import numpy as np
import pandas as pd
from predictor import Predictor
from walkforward import walkforward_full

class FormulaPredictor(Predictor):
    def train(self, features, target):
        pass

    def predict(self, features):
        g1 = features["Feature.1"]
        g2 = features["Feature.2"]
        g3 = features["Feature.3"]
        g4 = features["Feature.4"]
        g5 = features["Feature.5"]
        g6 = features["Feature.6"]

        def csrank(df):
            r = df.rank(axis=1)
            n = df.notna().sum(axis=1).values.reshape(-1, 1)
            return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

        # Test formula: just F1-F6 + F5-F6 + F3-F6 + F4-F6 (the aligned ones)
        # Actually, let's just test F1.
        pred = csrank(g1)
        
        # Smooth and demean
        pred = pred.ewm(span=720, min_periods=1).mean()
        pred = pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)
        return pred

if __name__ == "__main__":
    res = walkforward_full(FormulaPredictor, gauge_fix=False, verbose=False)
    print(f"Formula (F1) Full Sharpe: {res['sharpe']:.6f}")
