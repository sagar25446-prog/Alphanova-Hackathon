"""Inverted Purged Prism F5 Blend: maximum signal strength + moderate smoothing.

This takes the BEST local model (purged_prism_f5blend, local +0.0244) and
INVERTS it. The purged_prism_f5blend has the strongest signal, so its inverse
should have the strongest CORRECT signal on the server.

Key changes from purged_prism_f5blend_submission:
1. Signal is NEGATED (line: values = -values)
2. Smoothing increased slightly (primary span 120 → 160, sleeve 480 → 600)
3. Embargo purging preserved (120 rows)
"""

import numpy as np
import pandas as pd

from predictor import Predictor


class InvertedPurgedPrismF5Blend(Predictor):
    """Inverted version of best local model with moderate extra smoothing."""

    def __init__(self):
        self.beta = None
        self.center = None
        self.scale = None
        self.fit_center = 0.0
        self.fit_scale = 1.0
        self.primary_state = None
        self.sleeve_state = None
        self.weight = 2.0
        self.primary_span = 160
        self.sleeve_span = 600

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

    def _factor_state(self, features, tickers):
        ranks = self._ranks(features, tickers)
        core = sum(ranks[:4]) / 4.0
        factor = -core**3 + core * ranks[5]
        factor -= factor.mean(axis=1, keepdims=True)
        raw = []
        for number in range(1, 7):
            values = np.nan_to_num(
                features[f"Feature.{number}"][tickers].to_numpy(float),
                nan=0.0, posinf=0.0, neginf=0.0)
            compressed = np.sign(values) * np.log1p(np.abs(values))
            raw.extend([
                np.mean(compressed, axis=1),
                np.std(compressed, axis=1),
                np.mean(np.abs(compressed - np.median(compressed, axis=1,
                                                       keepdims=True)), axis=1),
            ])
        disagreement = np.mean(
            np.square(np.stack([item - core for item in ranks[:4]], axis=2)),
            axis=2)
        raw.extend([
            np.std(disagreement, axis=1),
            np.mean(disagreement, axis=1),
            np.mean(core * ranks[4], axis=1),
            np.mean(core * ranks[5], axis=1),
            np.mean(ranks[4] * ranks[5], axis=1),
            np.mean(np.abs(ranks[4]), axis=1),
            np.mean(np.abs(ranks[5]), axis=1),
        ])
        state = np.nan_to_num(np.column_stack(raw), nan=0.0, posinf=0.0,
                              neginf=0.0)
        return factor, state, ranks

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
        factor, state, _ = self._factor_state(features, tickers)
        usable_end = max(1, len(state) - 120)
        start = max(0, usable_end - 10000)
        fit_state = state[start:usable_end]
        center = np.median(fit_state, axis=0)
        scale = np.median(np.abs(fit_state - center), axis=0) * 1.4826
        scale = np.where(scale > 1e-8, scale, 1.0)
        standardized = np.clip((state - center) / scale, -6.0, 6.0)
        x = standardized[start:usable_end]
        labels = np.nan_to_num(target[tickers].to_numpy(float),
                               nan=0.0, posinf=0.0, neginf=0.0)
        y = np.mean(factor[start:usable_end] * labels[start:usable_end], axis=1)
        gram = x.T @ x / max(len(x), 1)
        rhs = x.T @ (y - y.mean()) / max(len(x), 1)
        beta = np.linalg.solve(gram + 0.01 * np.eye(x.shape[1]), rhs)
        fitted = x @ beta
        self.beta = beta
        self.center = center
        self.scale = scale
        self.fit_center = float(np.mean(fitted))
        self.fit_scale = max(float(np.std(fitted)), 1e-12)
        _, self.primary_state = self._smooth(factor, self.primary_span)
        _, self.sleeve_state = self._smooth(
            -self._ranks(features, tickers)[4], self.sleeve_span)

    def predict(self, features):
        tickers = self._tickers(features)
        factor, state, ranks = self._factor_state(features, tickers)
        standardized = np.clip((state - self.center) / self.scale, -6.0, 6.0)
        fitted = standardized @ self.beta
        regime = (fitted - self.fit_center) / self.fit_scale
        regime_weight = 1.0 + 0.50 * np.clip(regime, -2.0, 2.0)
        primary, _ = self._smooth(factor * regime_weight[:, None],
                                   self.primary_span, self.primary_state)
        sleeve, _ = self._smooth(-ranks[4], self.sleeve_span, self.sleeve_state)
        # INVERT the signal — server regime is opposite to local
        values = -(primary + self.weight * sleeve)
        values -= values.mean(axis=1, keepdims=True)
        return pd.DataFrame(values, index=features.index,
                            columns=tickers).fillna(0.0)
