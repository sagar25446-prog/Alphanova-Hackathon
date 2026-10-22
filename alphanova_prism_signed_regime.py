"""Bounded regime-orientation Prism signal.

The factor is the original causal Prism construction.  Unlike the prior
submissions, its training-only state response may smoothly change the final
sign through ``tanh``; the response is fit solely on the supplied training
prefix.  No city, identity, external, or future data is used.
"""

import numpy as np
import pandas as pd

from predictor import Predictor


class PrismSignedRegimePredictor(Predictor):
    """Low-capacity Prism with bounded state-dependent orientation."""

    def __init__(self):
        self.span = 120
        self.ridge = 0.01
        self.lookback = 10000
        self.orientation_strength = 0.75
        self.state_center = None
        self.state_scale = None
        self.beta = None
        self.fit_center = 0.0
        self.fit_scale = 1.0
        self.position_state = None

    def _tickers(self, features: pd.DataFrame) -> list[str]:
        return sorted(features["Feature.1"].columns, key=lambda name: int(name.split(".")[-1]))

    def _factor_and_state(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
        tickers = self._tickers(features)
        raw, ranks = [], []
        for number in range(1, 7):
            values = np.nan_to_num(features[f"Feature.{number}"][tickers].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
            raw.append(values)
            order = np.argsort(np.argsort(values, axis=1, kind="mergesort"), axis=1, kind="mergesort")
            ranks.append((order.astype(float) - 9.5) / 9.5)
        core = sum(ranks[:4]) / 4.0
        factor = -np.power(core, 3) + core * ranks[5]
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

    def _smooth(self, values: np.ndarray, initial: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        alpha = 2.0 / (self.span + 1.0)
        previous = np.zeros(values.shape[1], dtype=float) if initial is None else initial.copy()
        output = np.empty_like(values)
        for row in range(len(values)):
            previous = alpha * values[row] + (1.0 - alpha) * previous
            output[row] = previous
        output -= output.mean(axis=1, keepdims=True)
        return output, previous - previous.mean()

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        factor, state, tickers = self._factor_and_state(features)
        start = max(0, len(state) - self.lookback)
        x0 = state[start:]
        center = np.median(x0, axis=0)
        scale = np.median(np.abs(x0 - center), axis=0) * 1.4826
        scale = np.where(scale > 1e-8, scale, 1.0)
        x = np.clip((x0 - center) / scale, -6.0, 6.0)
        labels = np.nan_to_num(target[tickers].to_numpy(float)[start:], nan=0.0, posinf=0.0, neginf=0.0)
        response = np.mean(factor[start:] * labels, axis=1)
        gram = x.T @ x / max(len(x), 1)
        rhs = x.T @ (response - response.mean()) / max(len(x), 1)
        self.beta = np.linalg.solve(gram + self.ridge * np.eye(x.shape[1]), rhs)
        fitted = x @ self.beta
        self.state_center, self.state_scale = center, scale
        self.fit_center, self.fit_scale = float(fitted.mean()), max(float(fitted.std()), 1e-12)
        _, self.position_state = self._smooth(factor)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        factor, state, tickers = self._factor_and_state(features)
        x = np.clip((state - self.state_center) / self.state_scale, -6.0, 6.0)
        fitted = x @ self.beta
        z = (fitted - self.fit_center) / self.fit_scale
        orientation = np.tanh(self.orientation_strength * np.clip(z, -3.0, 3.0))
        values, _ = self._smooth(factor * orientation[:, None], self.position_state)
        result = pd.DataFrame(values, index=features.index, columns=tickers)
        return result.sub(result.mean(axis=1), axis=0).fillna(0.0)
