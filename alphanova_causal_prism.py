"""AlphaNova submission: causal conditional cross-sectional signal.

Before uploading, add your required participant details in the submission form
or the metadata location specified by the contest dashboard.
"""

import numpy as np
import pandas as pd

from predictor import Predictor


class CausalPrismPredictor(Predictor):
    """Low-capacity conditional signal with causal position smoothing.

    The base signal combines a tail-sensitive common component of Features 1–4
    with a cross-sectional interaction with Feature 6.  A ridge-regressed
    market-state multiplier is fitted only on the training slice.  It changes
    the signal's strength, not its per-asset identity, and all output rows are
    de-meaned.
    """

    def __init__(self):
        self.span = 120
        self.regime_strength = 0.50
        self.ridge = 0.01
        self.lookback = 10000
        self.interaction_weight = 1.00
        self.tickers = None
        self.state_center = None
        self.state_scale = None
        self.beta = None
        self.fit_center = 0.0
        self.fit_scale = 1.0
        self.position_state = None

    def _ticker_order(self, features: pd.DataFrame) -> list[str]:
        return sorted(
            features["Feature.1"].columns,
            key=lambda name: int(name.split(".")[-1]),
        )

    def _factor_and_state(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
        tickers = self._ticker_order(features)
        raw = []
        ranks = []
        for number in range(1, 7):
            values = features[f"Feature.{number}"][tickers].to_numpy(dtype=float)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            raw.append(values)
            ordered = pd.DataFrame(values).rank(axis=1, method="average").to_numpy(dtype=float)
            ranks.append((ordered - 10.5) / 9.5)

        core = sum(ranks[:4]) / 4.0
        factor = -np.power(core, 3) + self.interaction_weight * core * ranks[5]
        factor = factor - factor.mean(axis=1, keepdims=True)

        disagreement = np.mean(
            np.square(np.stack([item - core for item in ranks[:4]], axis=2)),
            axis=2,
        )
        blocks = []
        for values in raw:
            compressed = np.sign(values) * np.log1p(np.abs(values))
            blocks.extend([
                np.mean(compressed, axis=1),
                np.std(compressed, axis=1),
                np.mean(
                    np.abs(compressed - np.median(compressed, axis=1, keepdims=True)),
                    axis=1,
                ),
            ])
        blocks.extend([
            np.std(disagreement, axis=1),
            np.mean(disagreement, axis=1),
            np.mean(core * ranks[4], axis=1),
            np.mean(core * ranks[5], axis=1),
            np.mean(ranks[4] * ranks[5], axis=1),
            np.mean(np.abs(ranks[4]), axis=1),
            np.mean(np.abs(ranks[5]), axis=1),
        ])
        state = np.column_stack(blocks)
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        return factor, state, tickers

    def _smooth(self, values: np.ndarray, initial: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
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
        factor, state, tickers = self._factor_and_state(features)
        start = max(0, len(state) - self.lookback)
        train_state = state[start:]
        center = np.median(train_state, axis=0)
        scale = np.median(np.abs(train_state - center), axis=0) * 1.4826
        scale = np.where(scale > 1e-8, scale, 1.0)
        standardized = np.clip((state - center) / scale, -6.0, 6.0)
        x = standardized[start:]
        labels = target[tickers].to_numpy(dtype=float)
        labels = np.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.mean(factor[start:] * labels[start:], axis=1)
        gram = x.T @ x / max(len(x), 1)
        rhs = x.T @ (y - y.mean()) / max(len(x), 1)
        beta = np.linalg.solve(gram + self.ridge * np.eye(x.shape[1]), rhs)
        fitted = x @ beta
        fit_scale = float(np.std(fitted))

        _, position_state = self._smooth(factor)
        self.tickers = tickers
        self.state_center = center
        self.state_scale = scale
        self.beta = beta
        self.fit_center = float(np.mean(fitted))
        self.fit_scale = fit_scale if fit_scale > 1e-12 else 1.0
        self.position_state = position_state

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        factor, state, tickers = self._factor_and_state(features)
        if self.beta is None or self.state_center is None or self.state_scale is None:
            standardized = np.zeros_like(state)
            fitted = np.zeros(len(state), dtype=float)
            initial = None
        else:
            standardized = np.clip((state - self.state_center) / self.state_scale, -6.0, 6.0)
            fitted = standardized @ self.beta
            initial = self.position_state
        regime = (fitted - self.fit_center) / self.fit_scale
        weights = 1.0 + self.regime_strength * np.clip(regime, -2.0, 2.0)
        values, _ = self._smooth(factor * weights[:, None], initial)
        prediction = pd.DataFrame(values, index=features.index, columns=tickers)
        return prediction.sub(prediction.mean(axis=1), axis=0).fillna(0.0)
