import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
import numpy as np
import pandas as pd
from predictor import Predictor
from data_loader import load_period
from evaluation import backtest, sharpe_ratio

from demo_engineered import EngineeredXGBoostPredictor
from submission_xgb import TopRankPredictor

def calc_concentration(pred):
    # Concentration is usually calculated as sum of absolute weights or max weight?
    # Actually, concentration in this competition is something else, maybe Herfindahl-Hirschman Index (HHI)?
    # Or L2 norm of weights?
    # Let's just calculate L1 and L2 norm and variance.
    # Actually, the competition metric is "Concentration: L2 norm of portfolio weights" or similar.
    # Let's compute variance of cross-sectional weights.
    return (pred**2).sum(axis=1).mean()

d = load_period("031", "data/train", "small")
feat = d["features"]
tgt = d["target"]
ret = d["returns"]

n = len(feat)
n_train = int(n * 0.8)

print("Training Demo Engineered...")
p1 = EngineeredXGBoostPredictor()
p1.train(feat.iloc[:n_train], tgt.iloc[:n_train])
pred1 = p1.predict(feat.iloc[n_train:])

print("Training Submission XGB...")
p2 = TopRankPredictor()
p2.train(feat.iloc[:n_train], tgt.iloc[:n_train])
pred2 = p2.predict(feat.iloc[n_train:])

print("Demo Engineered Concentration proxy:", calc_concentration(pred1))
print("Submission XGB Concentration proxy:", calc_concentration(pred2))

# Also check max weight
print("Demo Max Weight:", pred1.abs().max(axis=1).mean())
print("Submission Max Weight:", pred2.abs().max(axis=1).mean())
