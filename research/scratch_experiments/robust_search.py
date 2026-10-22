"""Find the most STABLE signal: maximize min-period Sharpe, not average."""
import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
import numpy as np
import pandas as pd
from predictor import Predictor
from data_loader import list_periods, load_period
from evaluation import backtest
from walkforward import compute_val_rows

periods = sorted(list_periods("data/train"))

def eval_signal(predictor_fn, periods):
    """Evaluate a predictor across all periods, return per-period Sharpe & IC."""
    sharpes = []
    ics = []
    for p in periods:
        d = load_period(p, "data/train", "small")
        n = len(d["features"])
        n_val = compute_val_rows(n)
        n_train = n - n_val
        pred = predictor_fn(d["features"].iloc[n_train:])
        val_ret = d["returns"].iloc[n_train:]
        val_tgt = d["target"].iloc[n_train:]
        pf_ret = backtest(pred, val_ret)
        sh = float(pf_ret.mean() / pf_ret.std()) if pf_ret.std() > 0 else 0.0
        ic = np.nanmean(val_tgt.corrwith(pred, axis=1))
        sharpes.append(sh)
        ics.append(ic)
    return np.array(sharpes), np.array(ics)

def csrank(df):
    r = df.rank(axis=1)
    n = df.notna().sum(axis=1).values.reshape(-1, 1)
    return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

# Test many different signals
signals = {}

# 1. Individual negated features (which had positive Sharpe on last period)
for i in range(1, 7):
    for sign, label in [(1, f"+F{i}"), (-1, f"-F{i}")]:
        for span in [72, 240, 720]:
            def make_pred(sign=sign, i=i, span=span):
                def pred_fn(features):
                    f = features[f"Feature.{i}"]
                    p = (sign * f).rank(axis=1)
                    p = p.sub(p.mean(axis=1), axis=0).fillna(0.0)
                    p = p.ewm(span=span, min_periods=1).mean()
                    return p.sub(p.mean(axis=1), axis=0).fillna(0.0)
                return pred_fn
            signals[f"{label}_s{span}"] = make_pred()

# 2. Core combinations
for span in [72, 120, 240, 720]:
    # negcore3
    def make_nc3(span=span):
        def pred_fn(features):
            g = {j: features[f"Feature.{j}"] for j in range(1, 7)}
            ranks = [csrank(g[j]) for j in range(1, 7)]
            core = sum(ranks[:4]) / 4.0
            p = (-core**3).ewm(span=span, min_periods=1).mean()
            return p.sub(p.mean(axis=1), axis=0).fillna(0.0)
        return pred_fn
    signals[f"negcore3_s{span}"] = make_nc3()

    # primary
    def make_primary(span=span):
        def pred_fn(features):
            g = {j: features[f"Feature.{j}"] for j in range(1, 7)}
            ranks = [csrank(g[j]) for j in range(1, 7)]
            core = sum(ranks[:4]) / 4.0
            p = (-core**3 + core * ranks[5]).ewm(span=span, min_periods=1).mean()
            return p.sub(p.mean(axis=1), axis=0).fillna(0.0)
        return pred_fn
    signals[f"primary_s{span}"] = make_primary()

    # core only (linear)
    def make_core(span=span):
        def pred_fn(features):
            g = {j: features[f"Feature.{j}"] for j in range(1, 7)}
            ranks = [csrank(g[j]) for j in range(1, 7)]
            core = sum(ranks[:4]) / 4.0
            p = core.ewm(span=span, min_periods=1).mean()
            return p.sub(p.mean(axis=1), axis=0).fillna(0.0)
        return pred_fn
    signals[f"core_s{span}"] = make_core()

    # neg core (linear, negated)
    def make_negcore(span=span):
        def pred_fn(features):
            g = {j: features[f"Feature.{j}"] for j in range(1, 7)}
            ranks = [csrank(g[j]) for j in range(1, 7)]
            core = sum(ranks[:4]) / 4.0
            p = (-core).ewm(span=span, min_periods=1).mean()
            return p.sub(p.mean(axis=1), axis=0).fillna(0.0)
        return pred_fn
    signals[f"negcore_s{span}"] = make_negcore()

    # core * F5
    def make_core_f5(span=span):
        def pred_fn(features):
            g = {j: features[f"Feature.{j}"] for j in range(1, 7)}
            ranks = [csrank(g[j]) for j in range(1, 7)]
            core = sum(ranks[:4]) / 4.0
            p = (core * ranks[4]).ewm(span=span, min_periods=1).mean()
            return p.sub(p.mean(axis=1), axis=0).fillna(0.0)
        return pred_fn
    signals[f"core_f5_s{span}"] = make_core_f5()

    # core * F6
    def make_core_f6(span=span):
        def pred_fn(features):
            g = {j: features[f"Feature.{j}"] for j in range(1, 7)}
            ranks = [csrank(g[j]) for j in range(1, 7)]
            core = sum(ranks[:4]) / 4.0
            p = (core * ranks[5]).ewm(span=span, min_periods=1).mean()
            return p.sub(p.mean(axis=1), axis=0).fillna(0.0)
        return pred_fn
    signals[f"core_f6_s{span}"] = make_core_f6()

# 3. Blends: primary + negcore3
for w in [0.3, 0.5, 0.7]:
    for span in [72, 120, 240]:
        def make_blend(w=w, span=span):
            def pred_fn(features):
                g = {j: features[f"Feature.{j}"] for j in range(1, 7)}
                ranks = [csrank(g[j]) for j in range(1, 7)]
                core = sum(ranks[:4]) / 4.0
                primary = -core**3 + core * ranks[5]
                nc3 = -core**3
                p = (w * primary + (1 - w) * nc3).ewm(span=span, min_periods=1).mean()
                return p.sub(p.mean(axis=1), axis=0).fillna(0.0)
            return pred_fn
        signals[f"blend_{w:.1f}_s{span}"] = make_blend()

print(f"Testing {len(signals)} signal variants across {len(periods)} periods...")
print("=" * 100)
print(f"{'Signal':<25} {'MeanSh':>8} {'MinSh':>8} {'MaxSh':>8} {'StdSh':>8} {'NegPct':>8} {'MeanIC':>8} {'Score':>8}")
print("=" * 100)

results = []
for name, pred_fn in signals.items():
    sharpes, ics = eval_signal(pred_fn, periods)
    neg_pct = 100 * np.mean(sharpes < 0)
    # Score: we want high mean, low variance, and high min (robustness)
    # Use a "worst-case-aware" score: mean - 1*std (like a conservative Sharpe of Sharpes)
    score = np.mean(sharpes) - 1.0 * np.std(sharpes)
    results.append((name, np.mean(sharpes), np.min(sharpes), np.max(sharpes), 
                     np.std(sharpes), neg_pct, np.mean(ics), score))

# Sort by score (robust performance)
results.sort(key=lambda x: x[7], reverse=True)

for name, mean_sh, min_sh, max_sh, std_sh, neg_pct, mean_ic, score in results[:30]:
    print(f"{name:<25} {mean_sh:>8.4f} {min_sh:>8.4f} {max_sh:>8.4f} {std_sh:>8.4f} {neg_pct:>7.1f}% {mean_ic:>8.4f} {score:>8.4f}")

print("\n\nTop 10 by MIN Sharpe (worst case):")
results_by_min = sorted(results, key=lambda x: x[2], reverse=True)
for name, mean_sh, min_sh, max_sh, std_sh, neg_pct, mean_ic, score in results_by_min[:10]:
    print(f"{name:<25} {mean_sh:>8.4f} {min_sh:>8.4f} {max_sh:>8.4f} {std_sh:>8.4f} {neg_pct:>7.1f}% {mean_ic:>8.4f}")

print("\n\nTop 10 by NegPct (fewest losing periods):")
results_by_neg = sorted(results, key=lambda x: (x[5], -x[1]))
for name, mean_sh, min_sh, max_sh, std_sh, neg_pct, mean_ic, score in results_by_neg[:10]:
    print(f"{name:<25} {mean_sh:>8.4f} {min_sh:>8.4f} {max_sh:>8.4f} {std_sh:>8.4f} {neg_pct:>7.1f}% {mean_ic:>8.4f}")
