"""High-concentration stable signal with anti-dispersion properties.

Top performer #1 has:
- IC Dispersion = -0.39 (works BETTER in calm markets)
- Concentration = 0.26 (very stable direction)
- Compression Loss = -0.10

Strategy: Build a stable, high-concentration signal that weights DOWN 
during high-dispersion periods. Use ridge regression with dispersion 
conditioning.
"""

import numpy as np
import pandas as pd

from predictor import Predictor


class StableAntiDispersionPredictor(Predictor):
    """High-concentration signal with negative IC dispersion."""

    def __init__(self):
        self.beta = None
        self.center = None
        self.scale = None
        self.fit_center = 0.0
        self.fit_scale = 1.0
        self.position_state = None
        self.span = 200

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

    def _signal_and_state(self, features, tickers):
        ranks = self._ranks(features, tickers)
        core = sum(ranks[:4]) / 4.0

        # Primary signal: inverted prism core
        signal = -((-core**3 + core * ranks[5]))
        signal -= signal.mean(axis=1, keepdims=True)

        # State features for regime conditioning
        raw_vals = []
        for number in range(1, 7):
            values = np.nan_to_num(
                features[f"Feature.{number}"][tickers].to_numpy(float),
                nan=0.0, posinf=0.0, neginf=0.0)
            raw_vals.append(values)

        blocks = []
        for values in raw_vals:
            compressed = np.sign(values) * np.log1p(np.abs(values))
            blocks.extend([
                np.mean(compressed, axis=1),
                np.std(compressed, axis=1),
                np.mean(np.abs(compressed - np.median(compressed, axis=1,
                                                       keepdims=True)), axis=1),
            ])
        # Cross-sectional dispersion of each feature
        for values in raw_vals:
            blocks.append(np.std(values, axis=1))

        # Disagreement among features
        disagreement = np.mean(
            np.square(np.stack([item - core for item in ranks[:4]], axis=2)),
            axis=2)
        blocks.extend([
            np.std(disagreement, axis=1),
            np.mean(disagreement, axis=1),
            np.mean(core * ranks[4], axis=1),
            np.mean(core * ranks[5], axis=1),
            np.mean(ranks[4] * ranks[5], axis=1),
        ])

        # Add cross-sectional return dispersion proxy
        all_feat_std = np.std(np.stack(raw_vals, axis=2), axis=2)
        blocks.append(np.mean(all_feat_std, axis=1))
        blocks.append(np.std(all_feat_std, axis=1))

        state = np.nan_to_num(np.column_stack(blocks), nan=0.0, posinf=0.0,
                              neginf=0.0)
        return signal, state, ranks

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
        signal, state, _ = self._signal_and_state(features, tickers)
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
        # Use NEGATIVE target for inverted direction
        response = -np.mean(signal[start:usable_end] * labels[start:usable_end],
                           axis=1)
        gram = x.T @ x / max(len(x), 1)
        rhs = x.T @ (response - response.mean()) / max(len(x), 1)
        beta = np.linalg.solve(gram + 0.01 * np.eye(x.shape[1]), rhs)
        fitted = x @ beta
        self.beta = beta
        self.center = center
        self.scale = scale
        self.fit_center = float(np.mean(fitted))
        self.fit_scale = max(float(np.std(fitted)), 1e-12)
        _, self.position_state = self._smooth(signal, self.span)

    def predict(self, features):
        tickers = self._tickers(features)
        signal, state, _ = self._signal_and_state(features, tickers)
        standardized = np.clip((state - self.center) / self.scale, -6.0, 6.0)
        fitted = standardized @ self.beta
        regime = (fitted - self.fit_center) / self.fit_scale
        # Anti-dispersion: dampen signal in high-dispersion regimes
        weights = 1.0 + 0.50 * np.clip(regime, -2.0, 2.0)
        weighted_signal = signal * weights[:, None]
        values, _ = self._smooth(weighted_signal, self.span, self.position_state)
        values -= values.mean(axis=1, keepdims=True)
        return pd.DataFrame(values, index=features.index,
                            columns=tickers).fillna(0.0)
