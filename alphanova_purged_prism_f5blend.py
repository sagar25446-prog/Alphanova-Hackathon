"""Embargo-aware Prism with a fixed slow Feature-5 rank sleeve.

The primary Prism leg excludes the final 120 labelled rows from fitting.  A
second, unsupervised sleeve is a causal 480-row smoother of the cross-sectional
negative Feature-5 rank.  The fixed 2:1 sleeve weight was selected on the
released walk-forward panels before the final holdout comparison.
"""

import numpy as np
import pandas as pd

from predictor import Predictor


class _PrismLeg:
    def __init__(self):
        self.span = 120
        self.regime_strength = 0.50
        self.ridge = 0.01
        self.lookback = 10000
        self.state_center = None
        self.state_scale = None
        self.beta = None
        self.fit_center = 0.0
        self.fit_scale = 1.0
        self.position_state = None

    def _tickers(self, features):
        return sorted(features["Feature.1"].columns, key=lambda name: int(name.split(".")[-1]))

    def _factor_and_state(self, features):
        tickers = self._tickers(features)
        raw, ranks = [], []
        for number in range(1, 7):
            values = features[f"Feature.{number}"][tickers].to_numpy(dtype=float)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            raw.append(values)
            ordered = pd.DataFrame(values).rank(axis=1, method="average").to_numpy()
            ranks.append((ordered - 10.5) / 9.5)
        core = sum(ranks[:4]) / 4.0
        factor = -core**3 + core * ranks[5]
        factor -= factor.mean(axis=1, keepdims=True)
        disagreement = np.mean(np.square(np.stack([item - core for item in ranks[:4]], axis=2)), axis=2)
        blocks = []
        for values in raw:
            compressed = np.sign(values) * np.log1p(np.abs(values))
            blocks.extend([
                np.mean(compressed, axis=1),
                np.std(compressed, axis=1),
                np.mean(np.abs(compressed - np.median(compressed, axis=1, keepdims=True)), axis=1),
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
        state = np.nan_to_num(np.column_stack(blocks), nan=0.0, posinf=0.0, neginf=0.0)
        return factor, state, tickers

    def _smooth(self, values, initial=None):
        alpha = 2.0 / (self.span + 1.0)
        previous = np.zeros(values.shape[1], dtype=float) if initial is None else initial.copy()
        output = np.empty_like(values)
        for row in range(len(values)):
            previous = alpha * values[row] + (1.0 - alpha) * previous
            output[row] = previous
        output -= output.mean(axis=1, keepdims=True)
        return output, previous - previous.mean()

    def train(self, features, target):
        factor, state, tickers = self._factor_and_state(features)
        usable_end = max(1, len(state) - 120)
        start = max(0, usable_end - self.lookback)
        train_state = state[start:usable_end]
        center = np.median(train_state, axis=0)
        scale = np.median(np.abs(train_state - center), axis=0) * 1.4826
        scale = np.where(scale > 1e-8, scale, 1.0)
        standardized = np.clip((state - center) / scale, -6.0, 6.0)
        x = standardized[start:usable_end]
        labels = np.nan_to_num(target[tickers].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        y = np.mean(factor[start:usable_end] * labels[start:usable_end], axis=1)
        gram = x.T @ x / max(len(x), 1)
        rhs = x.T @ (y - y.mean()) / max(len(x), 1)
        self.beta = np.linalg.solve(gram + self.ridge * np.eye(x.shape[1]), rhs)
        fitted = x @ self.beta
        self.fit_center = float(np.mean(fitted))
        self.fit_scale = max(float(np.std(fitted)), 1e-12)
        _, self.position_state = self._smooth(factor)
        self.state_center = center
        self.state_scale = scale

    def predict(self, features):
        factor, state, tickers = self._factor_and_state(features)
        standardized = np.clip((state - self.state_center) / self.state_scale, -6.0, 6.0)
        fitted = standardized @ self.beta
        regime = (fitted - self.fit_center) / self.fit_scale
        weights = 1.0 + self.regime_strength * np.clip(regime, -2.0, 2.0)
        values, _ = self._smooth(factor * weights[:, None], self.position_state)
        return pd.DataFrame(values, index=features.index, columns=tickers)


class _F5Sleeve:
    def __init__(self):
        self.span = 480
        self.state = None

    def _rank_f5(self, features):
        tickers = sorted(features["Feature.5"].columns, key=lambda name: int(name.split(".")[-1]))
        values = np.nan_to_num(features["Feature.5"][tickers].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        order = np.argsort(np.argsort(values, axis=1, kind="mergesort"), axis=1, kind="mergesort")
        return (order.astype(float) - 9.5) / 9.5, tickers

    def _smooth(self, values, initial=None):
        alpha = 2.0 / (self.span + 1.0)
        state = np.zeros(values.shape[1], dtype=float) if initial is None else initial.copy()
        output = np.empty_like(values)
        for row in range(len(values)):
            state = alpha * values[row] + (1.0 - alpha) * state
            output[row] = state
        output -= output.mean(axis=1, keepdims=True)
        return output, state - state.mean()

    def train(self, features, target):
        rank, _ = self._rank_f5(features)
        _, self.state = self._smooth(-rank)

    def predict(self, features):
        rank, tickers = self._rank_f5(features)
        values, _ = self._smooth(-rank, self.state)
        return pd.DataFrame(values, index=features.index, columns=tickers)


class PurgedPrismF5Blend(Predictor):
    """Purged Prism plus a fixed 480-row negative-F5 rank sleeve."""

    def __init__(self):
        self.primary = _PrismLeg()
        self.sleeve = _F5Sleeve()
        self.weight = 2.0

    def train(self, features, target):
        self.primary.train(features, target)
        self.sleeve.train(features, target)

    def predict(self, features):
        first = self.primary.predict(features)
        second = self.sleeve.predict(features)[first.columns]
        values = first.to_numpy(dtype=float) + self.weight * second.to_numpy(dtype=float)
        values -= values.mean(axis=1, keepdims=True)
        return pd.DataFrame(values, index=features.index, columns=first.columns).fillna(0.0)
