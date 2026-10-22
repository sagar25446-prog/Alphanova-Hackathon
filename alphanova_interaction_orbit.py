"""Causal cross-sectional interaction signal for AlphaNova Season 1."""

import numpy as np
import pandas as pd

from predictor import Predictor


class InteractionOrbitPredictor(Predictor):
    """A low-capacity nonlinear factor with causal position smoothing."""

    def __init__(self):
        self.span = 120
        self.interaction_weight = 2.0
        self.position_state = None

    def _tickers(self, features: pd.DataFrame) -> list[str]:
        return sorted(
            features["Feature.1"].columns,
            key=lambda name: int(name.split(".")[-1]),
        )

    def _rank(self, values: np.ndarray) -> np.ndarray:
        clean = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        order = pd.DataFrame(clean).rank(axis=1, method="average").to_numpy(dtype=float)
        return (order - 10.5) / 9.5

    def _factor(self, features: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        tickers = self._tickers(features)
        ranks = []
        for number in range(1, 7):
            ranks.append(self._rank(features[f"Feature.{number}"][tickers].to_numpy(dtype=float)))
        core = sum(ranks[:4]) / 4.0
        values = -(core + self.interaction_weight * core * ranks[5])
        return values - values.mean(axis=1, keepdims=True), tickers

    def _smooth(
        self,
        values: np.ndarray,
        initial: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        alpha = 2.0 / (self.span + 1.0)
        previous = (
            np.zeros(values.shape[1], dtype=float)
            if initial is None
            else initial.copy()
        )
        output = np.empty_like(values)
        for row in range(len(values)):
            previous = alpha * values[row] + (1.0 - alpha) * previous
            output[row] = previous
        output = output - output.mean(axis=1, keepdims=True)
        return output, previous - previous.mean()

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        values, _ = self._factor(features)
        _, self.position_state = self._smooth(values)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        values, tickers = self._factor(features)
        prediction, _ = self._smooth(values, self.position_state)
        result = pd.DataFrame(prediction, index=features.index, columns=tickers)
        return result.sub(result.mean(axis=1), axis=0).fillna(0.0)
