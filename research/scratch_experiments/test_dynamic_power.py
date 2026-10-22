import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
import numpy as np
import pandas as pd
from predictor import Predictor

class TopRankPredictor(Predictor):
    def __init__(self):
        super().__init__()
        self.span = 120

    @staticmethod
    def _csrank(df):
        r = df.rank(axis=1)
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

    def train(self, features, target):
        pass

    def predict(self, features):
        feature_names = features.columns.get_level_values(0).unique()
        g = {f: self._csrank(features[f]) for f in feature_names}
        
        n_feats = len(feature_names)
        corrs = np.zeros((n_feats, n_feats))
        f_list = list(feature_names)
        
        for i in range(n_feats):
            v1 = g[f_list[i]].values.flatten()
            valid1 = ~np.isnan(v1)
            for j in range(i, n_feats):
                v2 = g[f_list[j]].values.flatten()
                valid = valid1 & ~np.isnan(v2)
                if valid.sum() > 0:
                    c = np.corrcoef(v1[valid], v2[valid])[0,1]
                else:
                    c = 0.0
                corrs[i, j] = c
                corrs[j, i] = c
                
        mean_abs_corr = np.mean(np.abs(corrs), axis=1)
        sorted_indices = np.argsort(mean_abs_corr)
        
        f6_name = f_list[sorted_indices[0]]
        f5_name = f_list[sorted_indices[1]]
        core_names = [f_list[idx] for idx in sorted_indices[2:]]
        
        core = sum(g[name] for name in core_names) / 4.0
        core = self._csrank(core)
        f6 = self._csrank(g[f6_name])
        
        # Base signal
        raw_pred = -(core ** 3) + core * f6
        
        # Increase concentration!
        # Instead of raw_pred, we apply a power to push weights to the extremes.
        raw_pred = np.sign(raw_pred) * (np.abs(raw_pred) ** 3)
        
        pred = raw_pred.ewm(span=self.span, min_periods=1).mean()
        expanding_mean = pred.shift(1).expanding(min_periods=1).mean()
        pred = (pred - expanding_mean).fillna(0.0)
        
        # Normalize sum abs to 1? The evaluator does this anyway.
        return pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)

from walkforward import walkforward_full
res = walkforward_full(TopRankPredictor, gauge_fix=False, verbose=False)
print(f"Sharpe with power 3: {res['sharpe']:.4f}")
