"""Causal shared-core contrarian signal for AlphaNova Season 1.

This is a deliberately small, target-independent follow-up to the Prism
submissions.  It uses only same-row cross-sectional ranks of Features 1--4 and
an ordinary trailing exponential smoother; no city data or external data is
read.
"""

import numpy as np
import pandas as pd

from predictor import Predictor


class CoreContrarian120Predictor(Predictor):
    """Contrarian average rank of the shared first-four-feature component."""

    def __init__(self):
        self.span = 120
        self.position_state = None

    def _tickers(self, features: pd.DataFrame) -> list[str]:
        return sorted(
            features["Feature.1"].columns,
            key=lambda name: int(name.split(".")[-1]),
        )

    def _factor(self, features: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        tickers = self._tickers(features)
        ranks = []
        for number in range(1, 5):
            values = features[f"Feature.{number}"][tickers].to_numpy(dtype=float)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            ordered = pd.DataFrame(values).rank(axis=1, method="average").to_numpy(dtype=float)
            ranks.append((ordered - 10.5) / 9.5)
        factor = -sum(ranks) / 4.0
        factor -= factor.mean(axis=1, keepdims=True)
        return factor, tickers

    def _smooth(
        self,
        values: np.ndarray,
        initial: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        alpha = 2.0 / (self.span + 1.0)
        state = (
            np.zeros(values.shape[1], dtype=float)
            if initial is None
            else initial.copy()
        )
        output = np.empty_like(values)
        for row in range(len(values)):
            state = alpha * values[row] + (1.0 - alpha) * state
            output[row] = state
        output -= output.mean(axis=1, keepdims=True)
        return output, state - state.mean()

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        # Fixed factor: labels are intentionally not consulted.
        values, _ = self._factor(features)
        _, self.position_state = self._smooth(values)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        values, tickers = self._factor(features)
        prediction, _ = self._smooth(values, self.position_state)
        result = pd.DataFrame(prediction, index=features.index, columns=tickers)
        return result.sub(result.mean(axis=1), axis=0).fillna(0.0)
