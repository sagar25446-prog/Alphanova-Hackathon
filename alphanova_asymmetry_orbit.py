"""Causal cross-sectional asymmetry signal for AlphaNova Season 1."""

import numpy as np
import pandas as pd

from predictor import Predictor


class AsymmetryOrbitPredictor(Predictor):
    """A low-turnover nonlinear cross-sectional factor.

    The signal measures a third-feature interaction between the squared-rank
    asymmetry of Features 1 and 2 and the rank of Feature 5.  It has no asset
    identity inputs and uses only the row supplied to it.  The trailing state
    in ``train`` is used solely to make the causal position smoother continuous
    when ``predict`` starts.
    """

    def __init__(self):
        self.span = 120
        self.position_state = None

    def _tickers(self, features: pd.DataFrame) -> list[str]:
        return sorted(
            features["Feature.1"].columns,
            key=lambda name: int(name.split(".")[-1]),
        )

    def _rank(self, values: np.ndarray) -> np.ndarray:
        clean = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        ordered = pd.DataFrame(clean).rank(axis=1, method="average").to_numpy(dtype=float)
        return (ordered - 10.5) / 9.5

    def _factor(self, features: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        tickers = self._tickers(features)
        first = self._rank(features["Feature.1"][tickers].to_numpy(dtype=float))
        second = self._rank(features["Feature.2"][tickers].to_numpy(dtype=float))
        fifth = self._rank(features["Feature.5"][tickers].to_numpy(dtype=float))
        values = (second * second - first * first) * fifth
        values = values - values.mean(axis=1, keepdims=True)
        return values, tickers

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
