"""Embargo-aware 240-row rank-trend ridge predictor.

The target is forward-looking, so the final 120 labelled training rows are
purged before fitting.  Features are contemporaneous cross-sectional ranks and
their causal 240-row exponentially weighted changes.  The predictor uses no
hidden evaluation data, city coordinates, external data, or ticker identity.
"""

import numpy as np
import pandas as pd

from predictor import Predictor


class Trend240RidgePredictor(Predictor):
    """Low-capacity ridge on slow causal feature-rank trends."""

    def __init__(self):
        self.trend_span = 240
        self.position_span = 120
        self.embargo = 120
        self.lookback = 5000
        self.ridge = 1.0
        self.beta = None
        self.rank_state = None
        self.position_state = None

    def _tickers(self, features: pd.DataFrame) -> list[str]:
        return sorted(
            features["Feature.1"].columns,
            key=lambda name: int(name.split(".")[-1]),
        )

    def _rank(self, values: np.ndarray) -> np.ndarray:
        clean = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        order = np.argsort(
            np.argsort(clean, axis=1, kind="mergesort"),
            axis=1,
            kind="mergesort",
        )
        ranked = (order.astype(float) - 9.5) / 9.5
        return ranked - ranked.mean(axis=1, keepdims=True)

    def _trend_features(
        self,
        features: pd.DataFrame,
        initial_state: list[np.ndarray] | None = None,
        save_state: bool = False,
    ) -> tuple[np.ndarray, list[str]]:
        tickers = self._tickers(features)
        alpha = 2.0 / (self.trend_span + 1.0)
        blocks = []
        states = []
        for offset, number in enumerate(range(1, 7)):
            values = features[f"Feature.{number}"][tickers].to_numpy(dtype=float)
            ranks = self._rank(values)
            state = (
                np.zeros(ranks.shape[1], dtype=float)
                if initial_state is None
                else initial_state[offset].copy()
            )
            trend = np.empty_like(ranks)
            for row in range(len(ranks)):
                state = alpha * ranks[row] + (1.0 - alpha) * state
                trend[row] = state - ranks[row]
            trend -= trend.mean(axis=1, keepdims=True)
            blocks.append(trend)
            states.append(state - state.mean())
        if save_state:
            self.rank_state = states
        return np.stack(blocks, axis=2), tickers

    def _smooth(
        self,
        values: np.ndarray,
        initial: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        alpha = 2.0 / (self.position_span + 1.0)
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
        design, tickers = self._trend_features(features, save_state=True)
        labels = np.nan_to_num(
            target[tickers].to_numpy(dtype=float),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        usable_end = max(1, len(design) - self.embargo)
        start = max(0, usable_end - self.lookback)
        x = design[start:usable_end].reshape(-1, design.shape[2])
        y = labels[start:usable_end].reshape(-1)
        scale = np.sqrt(np.mean(x * x, axis=0))
        scale = np.where(scale > 1e-8, scale, 1.0)
        xs = x / scale
        gram = xs.T @ xs / max(len(xs), 1)
        rhs = xs.T @ y / max(len(xs), 1)
        self.beta = np.linalg.solve(
            gram + self.ridge * np.eye(design.shape[2]),
            rhs,
        ) / scale

        fitted = np.einsum("tnk,k->tn", design, self.beta)
        _, self.position_state = self._smooth(fitted)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        design, tickers = self._trend_features(features, self.rank_state)
        raw = np.einsum("tnk,k->tn", design, self.beta)
        values, _ = self._smooth(raw, self.position_state)
        result = pd.DataFrame(values, index=features.index, columns=tickers)
        return result.sub(result.mean(axis=1), axis=0).fillna(0.0)
