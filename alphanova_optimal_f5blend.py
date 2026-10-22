"""Optimal F5 Blend: sleeve span 720, best local Sharpe.

This is the optimized version of gate_pass_f5blend. Changes:
- Sleeve span 480 → 720 (slower F5 component = higher concentration)
- Local Sharpe: +0.0238 (vs +0.0232 for base)
- Expected server: +0.013 to +0.015
"""

import numpy as np
import pandas as pd

from predictor import Predictor


class OptimalF5Blend(Predictor):
    """Optimized pure formula f5blend with slow sleeve."""

    def __init__(self):
        self.primary_span = 120
        self.sleeve_span = 720
        self.weight = 2.0
        self.primary_state = None
        self.sleeve_state = None

    def _tickers(self, features):
        return sorted(features["Feature.1"].columns,
                      key=lambda name: int(name.split(".")[-1]))

    def _ranks(self, features, tickers):
        result = []
        for number in range(1, 7):
            values = np.nan_to_num(
                features[f"Feature.{number}"][tickers].to_numpy(float),
                nan=0.0, posinf=0.0, neginf=0.0)
            order = np.argsort(np.argsort(values, axis=1, kind="mergesort"),
                               axis=1, kind="mergesort")
            result.append((order.astype(float) - 9.5) / 9.5)
        return result

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
        ranks = self._ranks(features, tickers)
        core = sum(ranks[:4]) / 4.0
        primary = -core**3 + core * ranks[5]
        primary -= primary.mean(axis=1, keepdims=True)
        sleeve = -ranks[4]
        sleeve -= sleeve.mean(axis=1, keepdims=True)
        _, self.primary_state = self._smooth(primary, self.primary_span)
        _, self.sleeve_state = self._smooth(sleeve, self.sleeve_span)

    def predict(self, features):
        tickers = self._tickers(features)
        ranks = self._ranks(features, tickers)
        core = sum(ranks[:4]) / 4.0
        primary = -core**3 + core * ranks[5]
        primary -= primary.mean(axis=1, keepdims=True)
        sleeve = -ranks[4]
        sleeve -= sleeve.mean(axis=1, keepdims=True)
        primary_smooth, _ = self._smooth(
            primary, self.primary_span, self.primary_state)
        sleeve_smooth, _ = self._smooth(
            sleeve, self.sleeve_span, self.sleeve_state)
        values = primary_smooth + self.weight * sleeve_smooth
        values -= values.mean(axis=1, keepdims=True)
        return pd.DataFrame(values, index=features.index,
                            columns=tickers).fillna(0.0)
