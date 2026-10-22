"""Pure F5 Sleeve Only: just smoothed -ranks[4].

Strip everything down to JUST the F5 signal. No prism core at all.
If F5 is the real alpha, this is the purest expression.
"""

import numpy as np
import pandas as pd

from predictor import Predictor


class PureF5SleevePredictor(Predictor):
    """Just the smoothed inverted Feature 5 rank. Nothing else."""

    def __init__(self):
        self.span = 360
        self.position_state = None

    def _tickers(self, features):
        return sorted(features["Feature.1"].columns,
                      key=lambda name: int(name.split(".")[-1]))

    def _smooth(self, values, span, initial=None):
        alpha = 2.0 / (span + 1.0)
        state = (np.zeros(values.shape[1], dtype=float)
                 if initial is None else initial.copy())
        output = np.empty_like(values)
        for row in range(len(values)):
            state = alpha * values[row] + (1.0 - alpha) * state
            output[row] = state
        output -= output.mean(axis=1, keepdims=True)
        return output, state - state.mean()

    def train(self, features, target):
        tickers = self._tickers(features)
        values = np.nan_to_num(
            features["Feature.5"][tickers].to_numpy(float),
            nan=0.0, posinf=0.0, neginf=0.0)
        order = np.argsort(np.argsort(values, axis=1, kind="mergesort"),
                           axis=1, kind="mergesort")
        ranks = (order.astype(float) - 9.5) / 9.5
        signal = -ranks
        signal -= signal.mean(axis=1, keepdims=True)
        _, self.position_state = self._smooth(signal, self.span)

    def predict(self, features):
        tickers = self._tickers(features)
        values = np.nan_to_num(
            features["Feature.5"][tickers].to_numpy(float),
            nan=0.0, posinf=0.0, neginf=0.0)
        order = np.argsort(np.argsort(values, axis=1, kind="mergesort"),
                           axis=1, kind="mergesort")
        ranks = (order.astype(float) - 9.5) / 9.5
        signal = -ranks
        signal -= signal.mean(axis=1, keepdims=True)
        result, _ = self._smooth(signal, self.span, self.position_state)
        return pd.DataFrame(result, index=features.index,
                            columns=tickers).fillna(0.0)
