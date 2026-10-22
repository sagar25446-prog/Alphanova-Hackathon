"""Hybrid Ridge: cross-sectional features + primary formula + heavy smoothing.

Combines Gemini Pro insights (Ridge regularization, EMA smoothing, cross-sectional 
z-scores, interactions) with our discovered primary formula signal, adapted for 
the actual MultiIndex column data format.
"""
import numpy as np
import pandas as pd
from predictor import Predictor


class HybridRidgePredictor(Predictor):
    """Ridge regression on rich cross-sectional features with heavy EMA smoothing."""

    def __init__(self):
        self.weights = None
        self.bias = 0.0
        self.alpha = 1000.0  # Very heavy L2 regularization
        self.span = 120      # EWM smoothing span

    @staticmethod
    def _csrank(df):
        """Cross-sectional rank normalized to [-1, 1]."""
        r = df.rank(axis=1)
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

    def _build_features(self, features):
        """Build rich cross-sectional feature matrix.
        
        Returns: X array (n_rows * n_tickers, n_features), tickers list, index
        """
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        tickers = list(g[1].columns)
        
        # Cross-sectional ranks
        ranks = {i: self._csrank(g[i]) for i in range(1, 7)}
        
        # Core features (cross-sectional ranks of each feature)
        feat_blocks = []
        for i in range(1, 7):
            feat_blocks.append(ranks[i][tickers].values.reshape(-1, 1))
        
        # Cross-sectional z-scores (alternative normalization)
        for i in range(1, 7):
            mu = g[i].mean(axis=1).values.reshape(-1, 1)
            sd = g[i].std(axis=1).values.reshape(-1, 1)
            sd = np.where(sd < 1e-8, 1e-8, sd)
            z = ((g[i][tickers].values - mu) / sd).reshape(-1, 1)
            feat_blocks.append(np.nan_to_num(z))
        
        # Difference features (like demo_engineered)
        diffs = [
            (1, 2), (3, 4), (1, 3), (1, 6), (5, 6),
        ]
        for a, b in diffs:
            d = self._csrank(g[a] - g[b])
            feat_blocks.append(d[tickers].values.reshape(-1, 1))
        
        # Primary formula components
        core = sum(ranks[i] for i in range(1, 5)) / 4.0
        negcore3 = (-core**3)
        primary = negcore3 + core * ranks[6]
        feat_blocks.append(core[tickers].values.reshape(-1, 1))
        feat_blocks.append(negcore3[tickers].values.reshape(-1, 1))
        feat_blocks.append(primary[tickers].values.reshape(-1, 1))
        
        # Nonlinear interactions (key pairs only to keep dimension low)
        for a, b in [(1, 6), (1, 5), (3, 6)]:
            inter = (ranks[a] * ranks[b])
            feat_blocks.append(inter[tickers].values.reshape(-1, 1))
        
        # Tanh of core ranks (soft clipping)
        for i in [1, 5, 6]:
            feat_blocks.append(np.tanh(ranks[i][tickers].values).reshape(-1, 1))
        
        X = np.hstack(feat_blocks)
        X = np.nan_to_num(X)
        return X, tickers, ranks[1].index

    def train(self, features, target):
        """Train Ridge regression on cross-sectional features."""
        X, tickers, _ = self._build_features(features)
        y = target[tickers].values.reshape(-1)
        
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        X_ok = X[ok]
        y_ok = y[ok]
        
        # Ridge closed-form: w = (X'X + alpha*I)^-1 X'y
        n_feat = X_ok.shape[1]
        XtX = X_ok.T @ X_ok
        Xty = X_ok.T @ y_ok
        self.weights = np.linalg.solve(XtX + self.alpha * np.eye(n_feat), Xty)
        self.bias = 0.0  # Features are already centered

    def predict(self, features):
        """Generate predictions with EWM smoothing and causal tilt removal."""
        X, tickers, index = self._build_features(features)
        
        raw = X @ self.weights + self.bias
        pred = pd.DataFrame(
            raw.reshape(-1, len(tickers)), index=index, columns=tickers
        )
        
        # EWM smoothing (critical for turnover reduction)
        pred = pred.ewm(span=self.span, min_periods=1).mean()
        
        # Causal tilt removal (from demo_engineered - removes static long/short bias)
        expanding_mean = pred.shift(1).expanding(min_periods=1).mean()
        pred = (pred - expanding_mean).fillna(0.0)
        
        # Cross-sectional de-mean
        pred = pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)
        return pred
