import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
import numpy as np
import pandas as pd
from predictor import Predictor
from walkforward import walkforward_full

class PrimaryPredictor(Predictor):
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
        
        pred = primary.ewm(span=720, min_periods=1).mean()
        pred = pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)
        return pred

if __name__ == "__main__":
    res = walkforward_full(PrimaryPredictor, gauge_fix=False, verbose=False)
    print(f"Primary Full Sharpe: {res['sharpe']:.6f}")
