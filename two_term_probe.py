"""Local compact ridge probe; not a submission."""

import numpy as np
import pandas as pd

from predictor import Predictor


class TwoTermRidgeProbe(Predictor):
    """Two selected rank interactions fitted with heavily shrunk ridge regression."""

    def __init__(self):
        self.ridge = 0.1
        self.span = 120
        self.lookback = 10000
        self.coef = None
        self.state = None

    def _design(self, features: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        tickers = sorted(features["Feature.1"].columns, key=lambda name: int(name.split(".")[-1]))
        ranks = []
        for number in range(1, 7):
            ordered = features[f"Feature.{number}"][tickers].rank(axis=1, method="average")
            ranks.append((ordered.to_numpy(dtype=float) - 10.5) / 9.5)
        core = sum(ranks[:4]) / 4.0
        terms = [
            -np.power(core, 3),
            core * ranks[5],
        ]
        return np.stack(terms, axis=2), tickers

    def _smooth(self, raw: np.ndarray, initial: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        alpha = 2.0 / (self.span + 1.0)
        state = np.zeros(raw.shape[1], dtype=float) if initial is None else initial.copy()
        output = np.empty_like(raw)
        for row in range(len(raw)):
            state = alpha * raw[row] + (1.0 - alpha) * state
            output[row] = state
        return output - output.mean(axis=1, keepdims=True), state - state.mean()

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        design, tickers = self._design(features)
        start = max(0, len(design) - self.lookback)
        x = design[start:].reshape(-1, design.shape[2])
        y = target[tickers].to_numpy(dtype=float)[start:].reshape(-1)
        scale = np.sqrt(np.mean(x * x, axis=0))
        scale = np.where(scale > 1e-8, scale, 1.0)
        standardized = x / scale
        gram = standardized.T @ standardized / max(len(x), 1)
        rhs = standardized.T @ y / max(len(x), 1)
        self.coef = np.linalg.solve(gram + self.ridge * np.eye(len(scale)), rhs) / scale
        raw = np.einsum("tnk,k->tn", design, self.coef)
        _, self.state = self._smooth(raw)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        design, tickers = self._design(features)
        raw = np.zeros(design.shape[:2]) if self.coef is None else np.einsum("tnk,k->tn", design, self.coef)
        values, _ = self._smooth(raw, self.state)
        return pd.DataFrame(values, index=features.index, columns=tickers)
