"""Inverted Purged Difference XGB.

The purged_difference_xgb has the LOWEST turnover (0.004) and second-best
server score (-0.0136). Inverting it should improve server Sharpe further.
"""

# /// script
# dependencies = [
#   "xgboost>=3.2.0",
# ]
# ///

import numpy as np
import pandas as pd
import xgboost as xgb

from predictor import Predictor


class InvertedPurgedDifferenceXGB(Predictor):
    """Inverted shallow XGBoost on difference-rank blocks."""

    def __init__(self):
        self.span = 120
        self.embargo = 120
        self.max_train_rows = 12000
        self.model = None
        self.position_state = None

    def _tickers(self, features):
        return sorted(features["Feature.1"].columns,
                      key=lambda name: int(name.split(".")[-1]))

    def _rank(self, values):
        clean = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        order = np.argsort(np.argsort(clean, axis=1, kind="mergesort"),
                           axis=1, kind="mergesort")
        return (order.astype(float) - 9.5) / 9.5

    def _design(self, features):
        tickers = self._tickers(features)
        raw = []
        for number in range(1, 7):
            values = features[f"Feature.{number}"][tickers].to_numpy(dtype=float)
            raw.append(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0))
        ranks = [self._rank(v) for v in raw]
        blocks = [
            ranks[5], ranks[0],
            self._rank(raw[0] - raw[1]),
            self._rank(raw[2] - raw[3]),
            self._rank(raw[0] - raw[2]),
            ranks[4],
        ]
        design = np.stack(blocks, axis=2)
        design -= design.mean(axis=1, keepdims=True)
        return np.nan_to_num(design), tickers

    def _smooth(self, values, initial=None):
        alpha = 2.0 / (self.span + 1.0)
        state = (np.zeros(values.shape[1], dtype=float)
                 if initial is None else initial.copy())
        output = np.empty_like(values)
        for row in range(len(values)):
            state = alpha * values[row] + (1.0 - alpha) * state
            output[row] = state
        output -= output.mean(axis=1, keepdims=True)
        return output, state - state.mean()

    def train(self, features, target):
        design, tickers = self._design(features)
        labels = np.nan_to_num(target[tickers].to_numpy(dtype=float),
                               nan=0.0, posinf=0.0, neginf=0.0)
        usable_end = max(1, len(design) - self.embargo)
        start = max(0, usable_end - self.max_train_rows)
        x = design[start:usable_end].reshape(-1, design.shape[2])
        y = labels[start:usable_end].reshape(-1)
        self.model = xgb.train(
            {
                "objective": "reg:squarederror",
                "max_depth": 2,
                "eta": 0.05,
                "subsample": 0.75,
                "colsample_bytree": 0.80,
                "min_child_weight": 1000.0,
                "reg_lambda": 20.0,
                "tree_method": "hist",
                "nthread": 4,
                "seed": 20260827,
                "verbosity": 0,
            },
            xgb.DMatrix(x, label=y),
            num_boost_round=30,
        )
        fitted = self.model.predict(
            xgb.DMatrix(design.reshape(-1, design.shape[2]))
        ).reshape(design.shape[:2])
        _, self.position_state = self._smooth(fitted)

    def predict(self, features):
        design, tickers = self._design(features)
        raw = self.model.predict(
            xgb.DMatrix(design.reshape(-1, design.shape[2]))
        ).reshape(design.shape[:2])
        values, _ = self._smooth(raw, self.position_state)
        # INVERT the signal
        values = -values
        result = pd.DataFrame(values, index=features.index, columns=tickers)
        return result.sub(result.mean(axis=1), axis=0).fillna(0.0)
