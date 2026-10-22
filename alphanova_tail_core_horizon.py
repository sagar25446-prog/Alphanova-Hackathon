"""Research candidate — do not upload before an official current-map novelty check."""

import numpy as np
import pandas as pd

from predictor import Predictor


class TailCoreHorizonPredictor(Predictor):
    """A low-capacity, stateless conditional cross-sectional tail signal.

    The signal uses contemporaneous ranks only.  A small ridge model fitted on
    the training target changes the factor's magnitude by current market state;
    it cannot choose asset identities and is never accessed in predict().
    """

    def __init__(self):
        self.span = 120
        self.regime_strength = 0.50
        self.ridge = 0.01
        self.lookback = 10000
        self.state_center = None
        self.state_scale = None
        self.beta = None
        self.fit_center = 0.0
        self.fit_scale = 1.0

    def _tickers(self, features: pd.DataFrame) -> list[str]:
        return sorted(features["Feature.1"].columns, key=lambda name: int(name.split(".")[-1]))

    def _factor_and_state(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
        tickers = self._tickers(features)
        raw = []
        ranks = []
        for number in range(1, 7):
            values = features[f"Feature.{number}"][tickers].to_numpy(dtype=float)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            raw.append(values)
            order = np.argsort(np.argsort(values, axis=1, kind="mergesort"), axis=1, kind="mergesort")
            ranks.append((order.astype(float) - 9.5) / 9.5)

        core = sum(ranks[:4]) / 4.0
        factor = -np.power(core, 3)
        factor = factor - factor.mean(axis=1, keepdims=True)
        disagreement = np.mean(
            np.square(np.stack([item - core for item in ranks[:4]], axis=2)), axis=2
        )
        state_blocks = []
        for values in raw:
            compressed = np.sign(values) * np.log1p(np.abs(values))
            state_blocks.extend([
                np.mean(compressed, axis=1),
                np.std(compressed, axis=1),
                np.mean(
                    np.abs(compressed - np.median(compressed, axis=1, keepdims=True)), axis=1
                ),
            ])
        state_blocks.extend([
            np.std(disagreement, axis=1),
            np.mean(disagreement, axis=1),
            np.mean(core * ranks[4], axis=1),
            np.mean(core * ranks[5], axis=1),
            np.mean(ranks[4] * ranks[5], axis=1),
            np.mean(np.abs(ranks[4]), axis=1),
            np.mean(np.abs(ranks[5]), axis=1),
        ])
        state = np.nan_to_num(np.column_stack(state_blocks), nan=0.0, posinf=0.0, neginf=0.0)
        return factor, state, tickers

    def _smooth(self, values: np.ndarray) -> np.ndarray:
        alpha = 2.0 / (self.span + 1.0)
        previous = np.zeros(values.shape[1], dtype=float)
        output = np.empty_like(values)
        for row in range(len(values)):
            previous = alpha * values[row] + (1.0 - alpha) * previous
            output[row] = previous
        return output - output.mean(axis=1, keepdims=True)

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        factor, state, tickers = self._factor_and_state(features)
        start = max(0, len(state) - self.lookback)
        train_state = state[start:]
        center = np.median(train_state, axis=0)
        scale = np.median(np.abs(train_state - center), axis=0) * 1.4826
        scale = np.where(scale > 1e-8, scale, 1.0)
        standardized = np.clip((train_state - center) / scale, -6.0, 6.0)
        labels = np.nan_to_num(target[tickers].to_numpy(dtype=float)[start:], nan=0.0, posinf=0.0, neginf=0.0)
        response = np.mean(factor[start:] * labels, axis=1)
        gram = standardized.T @ standardized / max(len(standardized), 1)
        rhs = standardized.T @ (response - response.mean()) / max(len(standardized), 1)
        self.beta = np.linalg.solve(gram + self.ridge * np.eye(gram.shape[0]), rhs)
        fitted = standardized @ self.beta
        self.state_center = center
        self.state_scale = scale
        self.fit_center = float(fitted.mean())
        self.fit_scale = max(float(fitted.std()), 1e-12)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        factor, state, tickers = self._factor_and_state(features)
        if self.beta is None or self.state_center is None or self.state_scale is None:
            fitted = np.zeros(len(state), dtype=float)
        else:
            standardized = np.clip((state - self.state_center) / self.state_scale, -6.0, 6.0)
            fitted = standardized @ self.beta
        regime = (fitted - self.fit_center) / self.fit_scale
        weights = 1.0 + self.regime_strength * np.clip(regime, -2.0, 2.0)
        values = self._smooth(factor * weights[:, None])
        prediction = pd.DataFrame(values, index=features.index, columns=tickers)
        return prediction.sub(prediction.mean(axis=1), axis=0).fillna(0.0)
