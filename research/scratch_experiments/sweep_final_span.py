import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
sys.path.insert(0, r"C:\Users\Sagar Jain\.gemini\antigravity\brain\23c3aefc-23d4-4c83-bcf6-595cd4899e13\scratch")
import numpy as np
import pandas as pd
from predictor import Predictor
from walkforward import walkforward_full
from data_loader import load_period

class SmoothDiscreteSmooth(Predictor):
    def __init__(self):
        super().__init__()
        self.span_feat = 240
        self.top_n = 2
        self.span_final = 24

    @staticmethod
    def _csrank(df):
        r = df.rank(axis=1)
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

    def train(self, features, target):
        pass

    def predict(self, features):
        f_names = features.columns.get_level_values(0).unique()
        g = {f: self._csrank(features[f]) for f in f_names}
        
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
        
        # Smooth features
        core = core.ewm(span=self.span_feat, min_periods=1).mean()
        f6 = f6.ewm(span=self.span_feat, min_periods=1).mean()
        
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
                
        # Final Smoothing!
        pred = discrete_pred.ewm(span=self.span_final, min_periods=1).mean()
        
        return pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)

for span in [1, 2, 6, 12, 24, 48, 120, 240]:
    class P(SmoothDiscreteSmooth):
        def __init__(self):
            super().__init__()
            self.span_final = span
    res = walkforward_full(P, gauge_fix=False, verbose=False)
    d = load_period('031', 'data/train', 'small')
    pred = P().predict(d['features'])
    weights = pred.div(pred.abs().sum(axis=1), axis=0).fillna(0)
    conc = (weights**2).sum(axis=1).mean()
    print(f"Span {span:>3}: Sharpe = {res['sharpe']:.4f}, Concentration = {conc:.4f}")
