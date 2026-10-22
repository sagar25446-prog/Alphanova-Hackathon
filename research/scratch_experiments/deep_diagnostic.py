"""Deep diagnostic: understand why local +0.019 becomes server -0.024."""
import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
import numpy as np
import pandas as pd
from predictor import Predictor
from data_loader import list_periods, load_period
from evaluation import backtest, sharpe_ratio
from walkforward import compute_val_rows

class PrimaryPredictor(Predictor):
    def __init__(self, span=72):
        self.span = span
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
        pred = primary.ewm(span=self.span, min_periods=1).mean()
        pred = pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)
        return pred

periods = sorted(list_periods("data/train"))

print("=" * 90)
print(f"{'Period':>6} {'ValRows':>8} {'Sharpe':>10} {'IC_Tgt':>10} {'IC_Ret':>10} {'Turnover':>10} {'GrossSh':>10}")
print("=" * 90)

all_sharpes = []
all_ics = []

for p in periods:
    d = load_period(p, "data/train", "small")
    n = len(d["features"])
    n_val = compute_val_rows(n)
    n_train = n - n_val

    pred_obj = PrimaryPredictor(span=72)
    pred = pred_obj.predict(d["features"].iloc[n_train:])
    val_ret = d["returns"].iloc[n_train:]
    val_tgt = d["target"].iloc[n_train:]

    # Sharpe (net of TC)
    pf_ret = backtest(pred, val_ret)
    sh = float(pf_ret.mean() / pf_ret.std()) if pf_ret.std() > 0 else 0.0

    # IC with target
    ic_tgt = np.nanmean(val_tgt.corrwith(pred, axis=1))

    # IC with returns (predictive: pred_{t-1} vs ret_t)
    pred_shifted = pred.shift(1)
    ic_ret = np.nanmean(val_ret.corrwith(pred_shifted, axis=1))

    # Turnover
    turn = pred.shift(1).fillna(0).diff().abs().sum(axis=1).mean()

    # Gross sharpe
    gross = (pred.shift(1) * val_ret).sum(axis=1)
    gross_sh = float(gross.mean() / gross.std()) if gross.std() > 0 else 0.0

    all_sharpes.append(sh)
    all_ics.append(ic_tgt)

    print(f"{p:>6} {n_val:>8} {sh:>10.4f} {ic_tgt:>10.4f} {ic_ret:>10.4f} {turn:>10.4f} {gross_sh:>10.4f}")

print("=" * 90)
print(f"{'MEAN':>6} {'':>8} {np.mean(all_sharpes):>10.4f} {np.mean(all_ics):>10.4f}")
print(f"{'STD':>6} {'':>8} {np.std(all_sharpes):>10.4f} {np.std(all_ics):>10.4f}")
print(f"{'MIN':>6} {'':>8} {np.min(all_sharpes):>10.4f} {np.min(all_ics):>10.4f}")
print(f"{'MAX':>6} {'':>8} {np.max(all_sharpes):>10.4f} {np.max(all_ics):>10.4f}")
neg_count = sum(1 for s in all_sharpes if s < 0)
print(f"\nNegative Sharpe periods: {neg_count}/{len(all_sharpes)}")
print(f"Hit rate: {100*(1-neg_count/len(all_sharpes)):.1f}%")

# Also test NEGATED primary
print("\n\n" + "=" * 90)
print("NEGATED PRIMARY (span=72)")
print("=" * 90)

class NegPrimaryPredictor(Predictor):
    def __init__(self, span=72):
        self.span = span
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
        primary = core**3 - core * ranks[5]  # NEGATED
        pred = primary.ewm(span=self.span, min_periods=1).mean()
        pred = pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)
        return pred

neg_sharpes = []
neg_ics = []

for p in periods:
    d = load_period(p, "data/train", "small")
    n = len(d["features"])
    n_val = compute_val_rows(n)
    n_train = n - n_val

    pred_obj = NegPrimaryPredictor(span=72)
    pred = pred_obj.predict(d["features"].iloc[n_train:])
    val_ret = d["returns"].iloc[n_train:]
    val_tgt = d["target"].iloc[n_train:]

    pf_ret = backtest(pred, val_ret)
    sh = float(pf_ret.mean() / pf_ret.std()) if pf_ret.std() > 0 else 0.0
    ic_tgt = np.nanmean(val_tgt.corrwith(pred, axis=1))
    neg_sharpes.append(sh)
    neg_ics.append(ic_tgt)

print(f"{'MEAN':>6} {'':>8} {np.mean(neg_sharpes):>10.4f} {np.mean(neg_ics):>10.4f}")
print(f"Negative Sharpe periods: {sum(1 for s in neg_sharpes if s < 0)}/{len(neg_sharpes)}")

# Also test individual features
print("\n\n" + "=" * 90)
print("INDIVIDUAL FEATURE PREDICTORS (span=72, last period only)")
print("=" * 90)

d = load_period(periods[-1], "data/train", "small")
n = len(d["features"])
n_val = compute_val_rows(n)
n_train = n - n_val
val_feat = d["features"].iloc[n_train:]
val_ret = d["returns"].iloc[n_train:]
val_tgt = d["target"].iloc[n_train:]

for i in range(1, 7):
    for sign in [1, -1]:
        f = val_feat[f"Feature.{i}"]
        pred = (sign * f).rank(axis=1)
        pred = pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)
        pred = pred.ewm(span=72, min_periods=1).mean()
        pred = pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)

        pf_ret = backtest(pred, val_ret)
        sh = float(pf_ret.mean() / pf_ret.std()) if pf_ret.std() > 0 else 0.0
        ic = np.nanmean(val_tgt.corrwith(pred, axis=1))
        sign_str = "+" if sign == 1 else "-"
        print(f"  {sign_str}F{i}: Sharpe={sh:+.4f}, IC_Target={ic:+.4f}")

# Test negcore3 (which was our best before)
print("\n\n" + "=" * 90)
print("NEGCORE3 PER PERIOD")
print("=" * 90)

class NegCore3Predictor(Predictor):
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
        pred = (-core**3).ewm(span=720, min_periods=1).mean()
        pred = pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)
        return pred

nc3_sharpes = []
nc3_ics = []
for p in periods:
    d = load_period(p, "data/train", "small")
    n = len(d["features"])
    n_val = compute_val_rows(n)
    n_train = n - n_val
    pred_obj = NegCore3Predictor()
    pred = pred_obj.predict(d["features"].iloc[n_train:])
    val_ret = d["returns"].iloc[n_train:]
    val_tgt = d["target"].iloc[n_train:]
    pf_ret = backtest(pred, val_ret)
    sh = float(pf_ret.mean() / pf_ret.std()) if pf_ret.std() > 0 else 0.0
    ic = np.nanmean(val_tgt.corrwith(pred, axis=1))
    nc3_sharpes.append(sh)
    nc3_ics.append(ic)

print(f"NEGCORE3 MEAN: Sharpe={np.mean(nc3_sharpes):.4f}, IC={np.mean(nc3_ics):.4f}")
print(f"NEGCORE3 STD:  Sharpe={np.std(nc3_sharpes):.4f}")
print(f"NEGCORE3 MIN:  Sharpe={np.min(nc3_sharpes):.4f}")
neg_count = sum(1 for s in nc3_sharpes if s < 0)
print(f"Negative periods: {neg_count}/{len(nc3_sharpes)}")
