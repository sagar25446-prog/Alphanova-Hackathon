"""Causal diversified cross-sectional signal for AlphaNova Season 1."""

import numpy as np
import pandas as pd

from predictor import Predictor


class OrthogonalOrbitPredictor(Predictor):
    """A low-capacity conditional factor with a slow independent interaction leg."""

    def __init__(self):
        self.main_span = 120
        self.diversifier_span = 480
        self.blend_weight = 4.0
        self.regime_strength = 0.50
        self.ridge = 0.01
        self.lookback = 10000
        self.tickers = None
        self.state_center = None
        self.state_scale = None
        self.beta = None
        self.fit_center = 0.0
        self.fit_scale = 1.0
        self.main_state = None
        self.diversifier_state = None

    def _ticker_order(self, features: pd.DataFrame) -> list[str]:
        return sorted(
            features["Feature.1"].columns,
            key=lambda name: int(name.split(".")[-1]),
        )

    def _rank_panels(self, features: pd.DataFrame) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
        tickers = self._ticker_order(features)
        raw = []
        ranks = []
        for number in range(1, 7):
            values = features[f"Feature.{number}"][tickers].to_numpy(dtype=float)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            raw.append(values)
            ordered = pd.DataFrame(values).rank(axis=1, method="average").to_numpy(dtype=float)
            ranks.append((ordered - 10.5) / 9.5)
        return raw, ranks, tickers

    def _main_factor_and_state(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        raw, ranks, tickers = self._rank_panels(features)
        core = sum(ranks[:4]) / 4.0
        main_factor = -np.power(core, 3) + core * ranks[5]
        main_factor = main_factor - main_factor.mean(axis=1, keepdims=True)
        diversifier = core * ranks[5]
        diversifier = diversifier - diversifier.mean(axis=1, keepdims=True)
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
        state = np.nan_to_num(np.column_stack(blocks), nan=0.0, posinf=0.0, neginf=0.0)
        return main_factor, diversifier, state, tickers

    def _smooth(
        self,
        values: np.ndarray,
        span: int,
        initial: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        alpha = 2.0 / (span + 1.0)
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
        main_factor, diversifier, state, tickers = self._main_factor_and_state(features)
        start = max(0, len(state) - self.lookback)
        train_state = state[start:]
        center = np.median(train_state, axis=0)
        scale = np.median(np.abs(train_state - center), axis=0) * 1.4826
        scale = np.where(scale > 1e-8, scale, 1.0)
        standardized = np.clip((state - center) / scale, -6.0, 6.0)
        labels = np.nan_to_num(target[tickers].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        x = standardized[start:]
        y = np.mean(main_factor[start:] * labels[start:], axis=1)
        gram = x.T @ x / max(len(x), 1)
        rhs = x.T @ (y - y.mean()) / max(len(x), 1)
        beta = np.linalg.solve(gram + self.ridge * np.eye(x.shape[1]), rhs)
        fitted = x @ beta
        _, main_state = self._smooth(main_factor, self.main_span)
        _, diversifier_state = self._smooth(diversifier, self.diversifier_span)
        self.tickers = tickers
        self.state_center = center
        self.state_scale = scale
        self.beta = beta
        self.fit_center = float(np.mean(fitted))
        fitted_scale = float(np.std(fitted))
        self.fit_scale = fitted_scale if fitted_scale > 1e-12 else 1.0
        self.main_state = main_state
        self.diversifier_state = diversifier_state

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        main_factor, diversifier, state, tickers = self._main_factor_and_state(features)
        if self.beta is None or self.state_center is None or self.state_scale is None:
            fitted = np.zeros(len(state), dtype=float)
            main_initial = None
            diversifier_initial = None
        else:
            standardized = np.clip((state - self.state_center) / self.state_scale, -6.0, 6.0)
            fitted = standardized @ self.beta
            main_initial = self.main_state
            diversifier_initial = self.diversifier_state
        regime = (fitted - self.fit_center) / self.fit_scale
        main_weights = 1.0 + self.regime_strength * np.clip(regime, -2.0, 2.0)
        main_values, _ = self._smooth(main_factor * main_weights[:, None], self.main_span, main_initial)
        diversifier_values, _ = self._smooth(diversifier, self.diversifier_span, diversifier_initial)
        values = main_values + self.blend_weight * diversifier_values
        result = pd.DataFrame(values, index=features.index, columns=tickers)
        return result.sub(result.mean(axis=1), axis=0).fillna(0.0)
