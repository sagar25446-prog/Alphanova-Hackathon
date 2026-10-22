"""Local geometry probe; not a final submission."""

import numpy as np
import pandas as pd

from predictor import Predictor


class OrientationProbe(Predictor):
    """Measures whether an orthogonal, causal feature leg helps a core signal."""

    def __init__(self):
        self.span = 120
        self.weight = 4.00
        self.component = "core_f6"
        self.base = "cubic"
        self.state = None

    def _raw(self, features: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        tickers = sorted(features["Feature.1"].columns, key=lambda item: int(item.split(".")[-1]))
        ranks = []
        for number in range(1, 7):
            ordered = features[f"Feature.{number}"][tickers].rank(axis=1, method="average")
            ranks.append(((ordered - 10.5) / 9.5).to_numpy(dtype=float))
        core = sum(ranks[:4]) / 4.0
        if self.component == "f6":
            extra = ranks[5]
        elif self.component == "core_sq":
            extra = -np.square(core)
        elif self.component == "core_f6":
            extra = core * ranks[5]
        else:
            extra = core * ranks[4]
        base = -np.power(core, 3) if self.base == "cubic" else -core
        raw = base + self.weight * extra
        return raw - raw.mean(axis=1, keepdims=True), tickers

    def _smooth(self, raw: np.ndarray, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        alpha = 2.0 / (self.span + 1.0)
        prior = np.zeros(raw.shape[1], dtype=float) if state is None else state.copy()
        output = np.empty_like(raw)
        for row in range(len(raw)):
            prior = alpha * raw[row] + (1.0 - alpha) * prior
            output[row] = prior
        return output - output.mean(axis=1, keepdims=True), prior - prior.mean()

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        raw, _ = self._raw(features)
        _, self.state = self._smooth(raw)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        raw, tickers = self._raw(features)
        values, _ = self._smooth(raw, self.state)
        return pd.DataFrame(values, index=features.index, columns=tickers)
