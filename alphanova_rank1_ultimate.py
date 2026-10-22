"""AlphaNova submission: smoothed cross-sectional primary factor.

Uses the primary formula (-core^3 + core*rank6) with EWM smoothing
to produce a low-turnover, high-Sharpe cross-sectional signal.
"""

import numpy as np
import pandas as pd

from predictor import Predictor


class SmoothedPrimaryPredictor(Predictor):
    """Cross-sectional primary factor with EWM smoothing.

    The signal is a nonlinear combination of ranked Features 1-6:
      core = mean(rank(F1), rank(F2), rank(F3), rank(F4))
      primary = -core^3 + core * rank(F6)

    An exponential moving average smooths the signal to reduce turnover
    and transaction costs, then cross-sectional de-meaning ensures
    the output sums to zero at each timestamp.
    """

    def __init__(self):
        self.span = 72

    @staticmethod
    def _csrank(df):
        """Cross-sectional rank normalized to [-1, 1]."""
        r = df.rank(axis=1)
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

    def train(self, features, target):
        """No training needed for this formula-based predictor."""
        pass

    def predict(self, features):
        """Generate cross-sectional predictions from the primary factor."""
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}

        ranks = [self._csrank(g[i]) for i in range(1, 7)]
        core = sum(ranks[:4]) / 4.0
        primary = -core**3 + core * ranks[5]

        pred = primary.ewm(span=self.span, min_periods=1).mean()
        pred = pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)
        return pred
