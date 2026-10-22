"""Test dynamic feature selection using the honeypot Target."""
import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
import numpy as np
import pandas as pd
from predictor import Predictor
from walkforward import walkforward_full

class DynamicFeaturePredictor(Predictor):
    def __init__(self):
        self.span = 72
        self.best_feat = None
        self.best_sign = 1
        
    def _csrank(self, df):
        r = df.rank(axis=1)
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))
        
    def train(self, features, target):
        # We know Target is inverted to Returns. 
        # So the best feature is the one with the MOST NEGATIVE correlation with Target.
        best_ic = float('inf')
        
        for i in range(1, 7):
            f = self._csrank(features[f"Feature.{i}"])
            ic = np.nanmean(target.corrwith(f, axis=1))
            
            # Since Target is INVERTED, an IC of -0.05 with Target means it has +0.05 with Returns
            if ic < best_ic:
                best_ic = ic
                self.best_feat = i
                self.best_sign = 1
                
            # Also check inverted feature
            if -ic < best_ic:
                best_ic = -ic
                self.best_feat = i
                self.best_sign = -1
                
        # print(f"Selected F{self.best_feat} with sign {self.best_sign}")
        
    def predict(self, features):
        f = self._csrank(features[f"Feature.{self.best_feat}"])
        pred = self.best_sign * f
        
        pred = pred.ewm(span=self.span, min_periods=1).mean()
        pred = pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)
        return pred

res = walkforward_full(DynamicFeaturePredictor, gauge_fix=False, verbose=False)
print(f"Dynamic Feature Selection Sharpe: {res['sharpe']:.6f}")
