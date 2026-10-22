"""Slow-turnover inverse Prism experiment.

This is the inverse Prism that produced the best observed platform result,
with only its two causal smoothing spans changed from 120 to 360 rows.  The
change is intended to reduce the 5bp turnover drag; it uses no hidden or
external data.
"""

import numpy as np
import pandas as pd

from predictor import Predictor


class PrismInverseSlow360Predictor(Predictor):
    """Inverse Prism with a single predeclared slower-turnover setting."""

    def __init__(self):
        self.span = 360
        self.secondary_span = 360
        self.blend_weight = 0.25
        self.regime_strength = 0.50
        self.ridge = 0.01
        self.lookback = 10000
        self.state_center = None
        self.state_scale = None
        self.beta = None
        self.fit_center = 0.0
        self.fit_scale = 1.0
        self.position_state = None
        self.secondary_state = None

    def _tickers(self, features: pd.DataFrame) -> list[str]:
        return sorted(features["Feature.1"].columns, key=lambda name: int(name.split(".")[-1]))

    def _components(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        tickers = self._tickers(features)
        raw, ranks = [], []
        for number in range(1, 7):
            values = np.nan_to_num(
                features[f"Feature.{number}"][tickers].to_numpy(float),
                nan=0.0, posinf=0.0, neginf=0.0,
            )
            raw.append(values)
            order = np.argsort(np.argsort(values, axis=1, kind="mergesort"), axis=1, kind="mergesort")
            ranks.append((order.astype(float) - 9.5) / 9.5)
        core = sum(ranks[:4]) / 4.0
        primary = -np.power(core, 3) + core * ranks[5]
        primary -= primary.mean(axis=1, keepdims=True)
        secondary = -ranks[4]
        secondary -= secondary.mean(axis=1, keepdims=True)
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
        return primary, secondary, np.nan_to_num(np.column_stack(blocks)), tickers

    def _smooth(self, values: np.ndarray, span: int, initial: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        alpha = 2.0 / (span + 1.0)
        previous = np.zeros(values.shape[1], dtype=float) if initial is None else initial.copy()
        output = np.empty_like(values)
        for row in range(len(values)):
            previous = alpha * values[row] + (1.0 - alpha) * previous
            output[row] = previous
        output -= output.mean(axis=1, keepdims=True)
        return output, previous - previous.mean()

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        primary, secondary, state, tickers = self._components(features)
        start = max(0, len(state) - self.lookback)
        train_state = state[start:]
        center = np.median(train_state, axis=0)
        scale = np.median(np.abs(train_state - center), axis=0) * 1.4826
        scale = np.where(scale > 1e-8, scale, 1.0)
        standardized = np.clip((train_state - center) / scale, -6.0, 6.0)
        labels = np.nan_to_num(target[tickers].to_numpy(float)[start:], nan=0.0, posinf=0.0, neginf=0.0)
        response = np.mean(primary[start:] * labels, axis=1)
        gram = standardized.T @ standardized / max(len(standardized), 1)
        rhs = standardized.T @ (response - response.mean()) / max(len(standardized), 1)
        self.beta = np.linalg.solve(gram + self.ridge * np.eye(standardized.shape[1]), rhs)
        fitted = standardized @ self.beta
        self.state_center, self.state_scale = center, scale
        self.fit_center, self.fit_scale = float(fitted.mean()), max(float(fitted.std()), 1e-12)
        _, self.position_state = self._smooth(primary, self.span)
        _, self.secondary_state = self._smooth(secondary, self.secondary_span)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        primary, secondary, state, tickers = self._components(features)
        standardized = np.clip((state - self.state_center) / self.state_scale, -6.0, 6.0)
        fitted = standardized @ self.beta
        regime = (fitted - self.fit_center) / self.fit_scale
        weights = 1.0 + self.regime_strength * np.clip(regime, -2.0, 2.0)
        primary_values, _ = self._smooth(primary * weights[:, None], self.span, self.position_state)
        secondary_values, _ = self._smooth(secondary, self.secondary_span, self.secondary_state)
        values = -(primary_values + self.blend_weight * secondary_values)
        prediction = pd.DataFrame(values, index=features.index, columns=tickers)
        return prediction.sub(prediction.mean(axis=1), axis=0).fillna(0.0)
