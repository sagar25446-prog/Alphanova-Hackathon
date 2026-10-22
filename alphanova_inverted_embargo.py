"""Inverted Prism Five Tilt with embargo purging.

Same as the original prism_five_tilt_inverse but with:
1. Embargo purging (drop last 120 training rows)
2. Slightly heavier smoothing (span=180 instead of 120)
"""

import numpy as np
import pandas as pd

from predictor import Predictor


class InvertedPrismEmbargoPredictor(Predictor):
    """Inverted prism with embargo and moderate extra smoothing."""

    def __init__(self):
        self.span = 180
        self.secondary_span = 180
        self.blend_weight = 0.25
        self.regime_strength = 0.50
        self.ridge = 0.01
        self.lookback = 10000
        self.embargo = 120
        self.state_center = None
        self.state_scale = None
        self.beta = None
        self.fit_center = 0.0
        self.fit_scale = 1.0
        self.position_state = None
        self.secondary_state = None

    def _tickers(self, features):
        return sorted(features["Feature.1"].columns,
                      key=lambda name: int(name.split(".")[-1]))

    def _components(self, features):
        tickers = self._tickers(features)
        raw_vals, ranks = [], []
        for number in range(1, 7):
            values = np.nan_to_num(
                features[f"Feature.{number}"][tickers].to_numpy(float),
                nan=0.0, posinf=0.0, neginf=0.0)
            raw_vals.append(values)
            order = np.argsort(np.argsort(values, axis=1, kind="mergesort"),
                               axis=1, kind="mergesort")
            ranks.append((order.astype(float) - 9.5) / 9.5)
        core = sum(ranks[:4]) / 4.0
        primary = -np.power(core, 3) + core * ranks[5]
        primary -= primary.mean(axis=1, keepdims=True)
        secondary = -ranks[4]
        secondary -= secondary.mean(axis=1, keepdims=True)
        disagreement = np.mean(
            np.square(np.stack([item - core for item in ranks[:4]], axis=2)),
            axis=2)
        blocks = []
        for values in raw_vals:
            compressed = np.sign(values) * np.log1p(np.abs(values))
            blocks.extend([
                np.mean(compressed, axis=1),
                np.std(compressed, axis=1),
                np.mean(np.abs(compressed - np.median(compressed, axis=1,
                                                       keepdims=True)), axis=1),
            ])
        blocks.extend([
            np.std(disagreement, axis=1),
            np.mean(disagreement, axis=1),
            np.mean(core * ranks[4], axis=1),
            np.mean(core * ranks[5], axis=1),
            np.mean(ranks[4] * ranks[5], axis=1),
            np.mean(np.abs(ranks[4]), axis=1),
            np.mean(np.abs(ranks[5]), axis=1),
        ])
        return primary, secondary, np.nan_to_num(np.column_stack(blocks)), tickers

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
        primary, secondary, state, tickers = self._components(features)
        usable_end = max(1, len(state) - self.embargo)
        start = max(0, usable_end - self.lookback)
        train_state = state[start:usable_end]
        center = np.median(train_state, axis=0)
        scale = np.median(np.abs(train_state - center), axis=0) * 1.4826
        scale = np.where(scale > 1e-8, scale, 1.0)
        standardized = np.clip((train_state - center) / scale, -6.0, 6.0)
        labels = np.nan_to_num(
            target[tickers].to_numpy(float)[start:usable_end],
            nan=0.0, posinf=0.0, neginf=0.0)
        response = np.mean(primary[start:usable_end] * labels, axis=1)
        gram = standardized.T @ standardized / max(len(standardized), 1)
        rhs = standardized.T @ (response - response.mean()) / max(len(standardized), 1)
        self.beta = np.linalg.solve(
            gram + self.ridge * np.eye(standardized.shape[1]), rhs)
        fitted = standardized @ self.beta
        self.state_center, self.state_scale = center, scale
        self.fit_center = float(fitted.mean())
        self.fit_scale = max(float(fitted.std()), 1e-12)
        _, self.position_state = self._smooth(primary, self.span)
        _, self.secondary_state = self._smooth(secondary, self.secondary_span)

    def predict(self, features):
        primary, secondary, state, tickers = self._components(features)
        standardized = np.clip(
            (state - self.state_center) / self.state_scale, -6.0, 6.0)
        fitted = standardized @ self.beta
        regime = (fitted - self.fit_center) / self.fit_scale
        weights = 1.0 + self.regime_strength * np.clip(regime, -2.0, 2.0)
        primary_values, _ = self._smooth(
            primary * weights[:, None], self.span, self.position_state)
        secondary_values, _ = self._smooth(
            secondary, self.secondary_span, self.secondary_state)
        # INVERT
        values = -(primary_values + self.blend_weight * secondary_values)
        values -= values.mean(axis=1, keepdims=True)
        return pd.DataFrame(values, index=features.index,
                            columns=tickers).fillna(0.0)
