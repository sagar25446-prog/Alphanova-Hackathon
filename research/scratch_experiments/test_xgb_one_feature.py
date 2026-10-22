import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor
from walkforward import walkforward_full

class XGBOneFeature(Predictor):
    def __init__(self):
        self.model = None
        self.params = {
            "objective": "reg:squarederror",
            "max_depth": 2,
            "eta": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 1.0,
            "min_child_weight": 500,
            "reg_lambda": 5.0,
            "verbosity": 0,
            "nthread": 4,
        }
        self.n_rounds = 30
        self.span = 120

    @staticmethod
    def _csrank(df):
        r = df.rank(axis=1)
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

    def _extract_signal(self, features):
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
        
        # Base signal
        sig = -(core ** 3) + core * f6
        return self._csrank(sig)

    def _design(self, features):
        sig = self._extract_signal(features)
        tickers = list(sig.columns)
        X = sig.values.reshape(-1, 1)
        return X, tickers, sig.index

    def train(self, features, target):
        X, tickers, _ = self._design(features)
        y = target[tickers].values.reshape(-1)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        self.model = xgb.train(self.params, xgb.DMatrix(X[ok], label=y[ok]), num_boost_round=self.n_rounds)

    def predict(self, features):
        X, tickers, index = self._design(features)
        raw = self.model.predict(xgb.DMatrix(np.nan_to_num(X)))
        pred = pd.DataFrame(raw.reshape(-1, len(tickers)), index=index, columns=tickers)
        pred = pred.ewm(span=self.span, min_periods=1).mean()
        expanding_mean = pred.shift(1).expanding(min_periods=1).mean()
        pred = (pred - expanding_mean).fillna(0.0)
        pred = pred.sub(pred.mean(axis=1), axis=0)
        return pred.fillna(0.0)

print("Testing XGBoost with ONE dynamic feature:")
res = walkforward_full(XGBOneFeature, gauge_fix=False, verbose=False)
print(f"Overall Sharpe: {res['sharpe']:.4f}")
