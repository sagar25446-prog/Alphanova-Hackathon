"""Server-feedback orientation test of the simple shared-core factor.

The feature construction is identical to the locally positive
``CoreContrarian120`` benchmark.  Only the final orientation is reversed,
based on the observed sign flip between the original Prism submission and its
publicly evaluated inverse.  This file remains causal and target-independent.
"""

import numpy as np
import pandas as pd

from predictor import Predictor


class Core120ServerFlipPredictor(Predictor):
    """Low-turnover shared-core factor with fixed server-feedback sign."""

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
        # Positive orientation is the inverse of the locally profitable
        # contrarian benchmark, chosen only as a server A/B experiment.
        factor = sum(ranks) / 4.0
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
        values, _ = self._factor(features)
        _, self.position_state = self._smooth(values)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        values, tickers = self._factor(features)
        prediction, _ = self._smooth(values, self.position_state)
        result = pd.DataFrame(prediction, index=features.index, columns=tickers)
        return result.sub(result.mean(axis=1), axis=0).fillna(0.0)
