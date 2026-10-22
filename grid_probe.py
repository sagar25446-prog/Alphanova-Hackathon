"""Local causal response-surface probe; not a submission."""

import numpy as np
import pandas as pd

from predictor import Predictor


class RankGridProbe(Predictor):
    """Shrunk response surface over the common feature rank and Feature 6 rank."""

    def __init__(self):
        self.bins = 5
        self.shrinkage = 400.0
        self.span = 120
        self.grid = None
        self.state = None

    def _ranks(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
        tickers = sorted(features["Feature.1"].columns, key=lambda name: int(name.split(".")[-1]))
        ranks = []
        for number in range(1, 7):
            ordered = features[f"Feature.{number}"][tickers].rank(axis=1, method="average")
            ranks.append(((ordered.to_numpy(dtype=float) - 10.5) / 9.5))
        core = sum(ranks[:4]) / 4.0
        return core, ranks[5], tickers

    def _index(self, value: np.ndarray) -> np.ndarray:
        return np.clip(((value + 1.0) * self.bins / 2.0).astype(int), 0, self.bins - 1)

    def _smooth(self, raw: np.ndarray, prior: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        alpha = 2.0 / (self.span + 1.0)
        state = np.zeros(raw.shape[1], dtype=float) if prior is None else prior.copy()
        output = np.empty_like(raw)
        for row in range(len(raw)):
            state = alpha * raw[row] + (1.0 - alpha) * state
            output[row] = state
        return output - output.mean(axis=1, keepdims=True), state - state.mean()

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        core, feature_six, tickers = self._ranks(features)
        labels = target[tickers].to_numpy(dtype=float)
        left = self._index(core).reshape(-1)
        right = self._index(feature_six).reshape(-1)
        flat = left * self.bins + right
        sums = np.bincount(flat, weights=labels.reshape(-1), minlength=self.bins ** 2)
        counts = np.bincount(flat, minlength=self.bins ** 2)
        global_mean = float(np.mean(labels))
        table = (sums + self.shrinkage * global_mean) / (counts + self.shrinkage)
        self.grid = table.reshape(self.bins, self.bins)
        raw = self.grid[self._index(core), self._index(feature_six)]
        _, self.state = self._smooth(raw)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        core, feature_six, tickers = self._ranks(features)
        if self.grid is None:
            raw = np.zeros_like(core)
        else:
            raw = self.grid[self._index(core), self._index(feature_six)]
        values, _ = self._smooth(raw, self.state)
        return pd.DataFrame(values, index=features.index, columns=tickers)
