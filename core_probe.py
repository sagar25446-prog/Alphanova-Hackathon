"""Local structural-signal probe; not a final submission."""

import numpy as np
import pandas as pd

from predictor import Predictor


class StructuralCoreProbe(Predictor):
    """Slow contrarian common-component signal from the first four features."""

    def __init__(self):
        self.span = 50
        self.state = None

    def _raw_signal(self, features: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        tickers = sorted(features["Feature.1"].columns, key=lambda name: int(name.split(".")[-1]))
        ranks = []
        for number in range(1, 5):
            values = features[f"Feature.{number}"][tickers]
            rank = values.rank(axis=1, method="average")
            rank = (rank - 10.5) / 9.5
            ranks.append(rank.to_numpy(dtype=float))
        raw = -sum(ranks) / 4.0
        raw = raw - raw.mean(axis=1, keepdims=True)
        return raw, tickers

    def _smooth(self, raw: np.ndarray, initial: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        alpha = 2.0 / (self.span + 1.0)
        state = np.zeros(raw.shape[1], dtype=float) if initial is None else initial.copy()
        output = np.empty_like(raw)
        for row in range(len(raw)):
            state = alpha * raw[row] + (1.0 - alpha) * state
            output[row] = state
        output = output - output.mean(axis=1, keepdims=True)
        return output, state - state.mean()

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        raw, _ = self._raw_signal(features)
        _, self.state = self._smooth(raw)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        raw, tickers = self._raw_signal(features)
        values, self.state = self._smooth(raw, self.state)
        return pd.DataFrame(values, index=features.index, columns=tickers)
