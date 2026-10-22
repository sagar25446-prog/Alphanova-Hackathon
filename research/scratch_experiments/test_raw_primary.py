import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
import numpy as np
import pandas as pd
from predictor import Predictor
from walkforward import walkforward_full, walkforward_backtest

class RawPrimaryPredictor(Predictor):
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
        
        return primary.sub(primary.mean(axis=1), axis=0).fillna(0.0)

if __name__ == "__main__":
    res = walkforward_backtest(RawPrimaryPredictor, verbose=False)
    pred = res['predictions']
    r = res['returns']
    t = pd.read_parquet('data/train/031.small.target.parquet').reindex(pred.index)
    
    ic_t = np.nanmean(t.corrwith(pred, axis=1))
    print(f"IC Target: {ic_t:.4f}")
    
    gross = (pred.shift(1)*r).sum(axis=1)
    turn = pred.shift(1).fillna(0).diff().abs().sum(axis=1)
    net = gross - 0.0005*turn
    print(f"Gross Sharpe: {gross.mean()/gross.std():.4f}")
    print(f"Net Sharpe: {net.mean()/net.std():.4f}")
    print(f"Avg Turnover: {turn.mean():.4f}")
    
    full = walkforward_full(RawPrimaryPredictor, gauge_fix=False, verbose=False)
    print(f"Full Sharpe: {full['sharpe']:.6f}")
