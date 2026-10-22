"""Offline robustness probe for the original Prism architecture; not a submission."""

import runpy

import numpy as np
import pandas as pd

from alphanova_causal_prism import CausalPrismPredictor
from walkforward import walkforward_full


class StatelessPrismProbe(CausalPrismPredictor):
    """Restart the causal smoother for every inference frame.

    This intentionally removes the train-to-predict seed to measure whether the
    reported walk-forward edge is boundary-state dependent.
    """

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        factor, state, tickers = self._factor_and_state(features)
        if self.beta is None or self.state_center is None or self.state_scale is None:
            fitted = np.zeros(len(state), dtype=float)
        else:
            standardized = np.clip((state - self.state_center) / self.state_scale, -6.0, 6.0)
            fitted = standardized @ self.beta
        regime = (fitted - self.fit_center) / self.fit_scale
        weights = 1.0 + self.regime_strength * np.clip(regime, -2.0, 2.0)
        values, _ = self._smooth(factor * weights[:, None], initial=None)
        prediction = pd.DataFrame(values, index=features.index, columns=tickers)
        return prediction.sub(prediction.mean(axis=1), axis=0).fillna(0.0)


if __name__ == "__main__":
    result = walkforward_full(StatelessPrismProbe, gauge_fix=True, verbose=True)
    print(f"net_sharpe={result['sharpe']:+.6f}")
    print(f"city_proxy={result.get('city_novelty_deg', float('nan')):.2f}")
