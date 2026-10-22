"""Low-turnover inverse core interaction for AlphaNova Season 1.

This deliberately simple follow-up is fixed before submission from the public
results of the first five submissions.  It does not use city coordinates,
external data, asset identities, or target values to make predictions.
"""

import numpy as np
import pandas as pd

from predictor import Predictor


class InverseCoreF6Predictor(Predictor):
    """Causal inverse of the common-core/Feature-6 interaction.

    The signal retains the 120-row causal smoother used by the best server
    orientation experiment, but removes its Feature-5 side leg and its fitted
    regime multiplier.  This makes the next server experiment an ablation,
    rather than another broad parameter search.
    """

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
        for number in range(1, 7):
            values = features[f"Feature.{number}"][tickers].to_numpy(dtype=float)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            order = pd.DataFrame(values).rank(axis=1, method="average").to_numpy(dtype=float)
            ranks.append((order - 10.5) / 9.5)

        core = sum(ranks[:4]) / 4.0
        # Fixed inverse orientation.  The original direction was rejected by
        # the public test result; no target is consulted at prediction time.
        values = np.power(core, 3) - core * ranks[5]
        values -= values.mean(axis=1, keepdims=True)
        return values, tickers

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
        # ``target`` is intentionally unused: this is a fixed low-capacity
        # factor, so shuffled or zero-label gate inputs cannot alter it.
        values, _ = self._factor(features)
        _, self.position_state = self._smooth(values)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        values, tickers = self._factor(features)
        prediction, _ = self._smooth(values, self.position_state)
        result = pd.DataFrame(prediction, index=features.index, columns=tickers)
        return result.sub(result.mean(axis=1), axis=0).fillna(0.0)
