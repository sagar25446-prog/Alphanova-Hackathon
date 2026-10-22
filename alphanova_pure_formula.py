"""Pure formula signal: maximum concentration, minimum turnover.

Top performers have Concentration 0.26-0.37 (mine: 0.15).
Strategy: Use the simplest possible STABLE formula signal, inverted,
with EXTREME smoothing. No ML model at all — just a formula.
The stability gives high concentration and low turnover.
"""

import numpy as np
import pandas as pd

from predictor import Predictor


class PureFormulaPredictor(Predictor):
    """No ML, just a smoothed inverted rank formula."""

    def __init__(self):
        self.span = 360
        self.position_state = None

    def _tickers(self, features):
        return sorted(features["Feature.1"].columns,
                      key=lambda name: int(name.split(".")[-1]))

    def _signal(self, features, tickers):
        raw = []
        ranks = []
        for number in range(1, 7):
            values = np.nan_to_num(
                features[f"Feature.{number}"][tickers].to_numpy(float),
                nan=0.0, posinf=0.0, neginf=0.0)
            raw.append(values)
            order = np.argsort(np.argsort(values, axis=1, kind="mergesort"),
                               axis=1, kind="mergesort")
            ranks.append((order.astype(float) - 9.5) / 9.5)

        core = sum(ranks[:4]) / 4.0
        # Inverted prism formula
        signal = -(-core**3 + core * ranks[5])
        signal -= signal.mean(axis=1, keepdims=True)
        return signal

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
        signal = self._signal(features, tickers)
        _, self.position_state = self._smooth(signal, self.span)

    def predict(self, features):
        tickers = self._tickers(features)
        signal = self._signal(features, tickers)
        values, _ = self._smooth(signal, self.span, self.position_state)
        return pd.DataFrame(values, index=features.index,
                            columns=tickers).fillna(0.0)
