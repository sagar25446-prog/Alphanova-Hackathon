"""Rejected research candidate — do not upload."""

import numpy as np
import pandas as pd

from predictor import Predictor


class CoreAnchorPredictor(Predictor):
    """Low-turnover contrarian common-component cross-sectional signal.

    The train target is used only to select the orientation of one fixed,
    anonymous-feature formula.  Positions use a strictly trailing recurrence.
    """

    def __init__(self):
        self.span = 120
        self.orientation = 1.0
        self.position_state = None

    def _tickers(self, features: pd.DataFrame) -> list[str]:
        return sorted(features["Feature.1"].columns, key=lambda name: int(name.split(".")[-1]))

    def _raw_factor(self, features: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        tickers = self._tickers(features)
        ranks = []
        for number in range(1, 5):
            values = features[f"Feature.{number}"][tickers].to_numpy(dtype=float)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            order = np.argsort(np.argsort(values, axis=1, kind="mergesort"), axis=1, kind="mergesort")
            ranks.append((order.astype(float) - 9.5) / 9.5)
        core = sum(ranks) / 4.0
        values = -core
        return values - values.mean(axis=1, keepdims=True), tickers

    def _smooth(self, values: np.ndarray, initial: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        alpha = 2.0 / (self.span + 1.0)
        previous = np.zeros(values.shape[1], dtype=float) if initial is None else initial.copy()
        output = np.empty_like(values)
        for row in range(len(values)):
            previous = alpha * values[row] + (1.0 - alpha) * previous
            output[row] = previous
        output = output - output.mean(axis=1, keepdims=True)
        return output, previous - previous.mean()

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        factor, tickers = self._raw_factor(features)
        labels = np.nan_to_num(target[tickers].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        # Choose orientation from the *traded*, causally smoothed training
        # position rather than the unsmoothed formula.  The one-step timing and
        # smoothing are part of the model and therefore must be represented in
        # the training-side association as well.
        smoothed, state = self._smooth(factor)
        association = float(np.mean(smoothed * labels))
        self.orientation = 1.0 if association >= 0.0 else -1.0
        self.position_state = self.orientation * state

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        factor, tickers = self._raw_factor(features)
        values, _ = self._smooth(self.orientation * factor, self.position_state)
        prediction = pd.DataFrame(values, index=features.index, columns=tickers)
        return prediction.sub(prediction.mean(axis=1), axis=0).fillna(0.0)
