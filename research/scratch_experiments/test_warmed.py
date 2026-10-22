import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
import numpy as np
import pandas as pd
from predictor import Predictor
from walkforward import walkforward_full

class SmoothFeaturesPredictorWarmed(Predictor):
    def __init__(self):
        super().__init__()
        self.span = 240
        self.top_n = 2
        self.warmup_features = None

    @staticmethod
    def _csrank(df):
        r = df.rank(axis=1)
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

    def train(self, features, target):
        self.warmup_features = features.iloc[-self.span * 2:]

    def predict(self, features):
        if self.warmup_features is not None:
            f = pd.concat([self.warmup_features, features])
            n_warm = len(self.warmup_features)
        else:
            f = features
            n_warm = 0
            
        f_names = f.columns.get_level_values(0).unique()
        g = {f_name: self._csrank(f[f_name]) for f_name in f_names}
        
        n_feats = len(f_names)
        corrs = np.zeros((n_feats, n_feats))
        f_list = list(f_names)
        
        for i in range(n_feats):
            v1 = g[f_list[i]].values.flatten()
            valid1 = ~np.isnan(v1)
            for j in range(i, n_feats):
                v2 = g[f_list[j]].values.flatten()
                valid = valid1 & ~np.isnan(v2)
                c = np.corrcoef(v1[valid], v2[valid])[0,1] if valid.sum() > 0 else 0
                corrs[i, j] = c
                corrs[j, i] = c
                
        mean_abs_corr = np.mean(np.abs(corrs), axis=1)
        s = np.argsort(mean_abs_corr)
        
        f6_name = f_list[s[0]]
        core_names = [f_list[idx] for idx in s[2:]]
        
        core = sum(g[name] for name in core_names) / 4.0
        core = self._csrank(core)
        f6 = self._csrank(g[f6_name])
        
        core = core.ewm(span=self.span, min_periods=1).mean()
        f6 = f6.ewm(span=self.span, min_periods=1).mean()
        
        raw_pred = -(core ** 3) + core * f6
        
        ranks = raw_pred.rank(axis=1)
        n_assets = raw_pred.notna().sum(axis=1)
        
        discrete_pred = pd.DataFrame(0.0, index=raw_pred.index, columns=raw_pred.columns)
        for i in range(len(raw_pred)):
            r = ranks.iloc[i]
            n = n_assets.iloc[i]
            if n > 0:
                discrete_pred.iloc[i, r >= (n - self.top_n + 1)] = 1.0
                discrete_pred.iloc[i, r <= self.top_n] = -1.0
                
        # Drop the warmup rows!
        res = discrete_pred.iloc[n_warm:]
        return res.sub(res.mean(axis=1), axis=0).fillna(0.0)

print("Testing Smooth FEATURES (PRE-WARMED) -> Discrete:")
res = walkforward_full(SmoothFeaturesPredictorWarmed, gauge_fix=False, verbose=False)
print(f"Overall Sharpe: {res['sharpe']:.4f}")
