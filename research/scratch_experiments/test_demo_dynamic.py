import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor
from walkforward import walkforward_full

class XGBDynamicDemo(Predictor):
    def __init__(self):
        self.model = None
        self.params = {
            "objective": "reg:squarederror",
            "max_depth": 2,
            "eta": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
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

    def _blocks(self, features):
        f_names = features.columns.get_level_values(0).unique()
        g_raw = {f: self._csrank(features[f]) for f in f_names}
        
        n_feats = len(f_names)
        corrs = np.zeros((n_feats, n_feats))
        f_list = list(f_names)
        
        for i in range(n_feats):
            v1 = g_raw[f_list[i]].values.flatten()
            valid1 = ~np.isnan(v1)
            for j in range(i, n_feats):
                v2 = g_raw[f_list[j]].values.flatten()
                valid = valid1 & ~np.isnan(v2)
                c = np.corrcoef(v1[valid], v2[valid])[0,1] if valid.sum() > 0 else 0
                corrs[i, j] = c
                corrs[j, i] = c
                
        mean_abs_corr = np.mean(np.abs(corrs), axis=1)
        s = np.argsort(mean_abs_corr)
        
        # F6 is the most orthogonal, F5 is second
        f6 = g_raw[f_list[s[0]]]
        f5 = g_raw[f_list[s[1]]]
        
        # F1-F4 are the highly collinear ones. We don't know which is which.
        # But we know they are interchangeable mostly. Let's just use them arbitrarily.
        core1 = g_raw[f_list[s[2]]]
        core2 = g_raw[f_list[s[3]]]
        core3 = g_raw[f_list[s[4]]]
        core4 = g_raw[f_list[s[5]]]
        
        b = {}
        b["a"] = f6
        b["m"] = core1
        b["p"] = self._csrank(core1 - core2)
        b["q"] = self._csrank(core3 - core4)
        b["r"] = self._csrank(core1 - core3)
        b["s"] = f5
        return b

    def _design(self, features):
        b = self._blocks(features)
        tickers = list(b[next(iter(b))].columns)
        X = np.hstack([d[tickers].values.reshape(-1, 1) for d in b.values()])
        return X, tickers, b[next(iter(b))].index

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

print("Testing Dynamic Demo XGBoost:")
res = walkforward_full(XGBDynamicDemo, gauge_fix=False, verbose=False)
print(f"Overall Sharpe: {res['sharpe']:.4f}")
