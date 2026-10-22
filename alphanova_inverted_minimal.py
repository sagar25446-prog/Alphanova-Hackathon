"""Inverted Minimal Prism: simplest possible model, maximum smoothing.

Design: Take the causal_prism core factor, invert it, smooth it to death.
No secondary sleeve, no regime gating. Just pure inverted signal + 
ultra-heavy smoothing to minimize turnover.
"""

import numpy as np
import pandas as pd

from predictor import Predictor


class InvertedMinimalPrism(Predictor):
    """Minimal inverted core factor with extreme smoothing."""

    def __init__(self):
        self.span = 480
        self.position_state = None

    def _tickers(self, features):
        return sorted(features["Feature.1"].columns,
                      key=lambda name: int(name.split(".")[-1]))

    def _factor(self, features, tickers):
        ranks = []
        for number in range(1, 7):
            values = np.nan_to_num(
                features[f"Feature.{number}"][tickers].to_numpy(float),
                nan=0.0, posinf=0.0, neginf=0.0)
            order = np.argsort(np.argsort(values, axis=1, kind="mergesort"),
                               axis=1, kind="mergesort")
            ranks.append((order.astype(float) - 9.5) / 9.5)
        core = sum(ranks[:4]) / 4.0
        factor = -np.power(core, 3) + core * ranks[5]
        factor -= factor.mean(axis=1, keepdims=True)
        return factor

    def _smooth(self, values, span, initial=None):
        alpha = 2.0 / (span + 1.0)
        previous = (np.zeros(values.shape[1], dtype=float)
                     if initial is None else initial.copy())
        output = np.empty_like(values)
        for row in range(len(values)):
            previous = alpha * values[row] + (1.0 - alpha) * previous
            output[row] = previous
        output -= output.mean(axis=1, keepdims=True)
        return output, previous - previous.mean()

    def train(self, features, target):
        tickers = self._tickers(features)
        factor = self._factor(features, tickers)
        _, self.position_state = self._smooth(factor, self.span)

    def predict(self, features):
        tickers = self._tickers(features)
        factor = self._factor(features, tickers)
        smoothed, _ = self._smooth(factor, self.span, self.position_state)
        # INVERT — server regime is opposite
        values = -smoothed
        return pd.DataFrame(values, index=features.index,
                            columns=tickers).fillna(0.0)
