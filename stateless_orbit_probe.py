"""Offline only: overlap-safe inference probe for the diversified architecture."""

import numpy as np
import pandas as pd

from alphanova_orthogonal_orbit_v2 import OrthogonalOrbitV2Predictor
from walkforward import walkforward_full


class StatelessOrbitProbe(OrthogonalOrbitV2Predictor):
    """Research subclass that restarts causal filters for every inference frame."""

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        main, diversifier, state, tickers = self._components(features)
        if self.beta is None or self.state_center is None or self.state_scale is None:
            fitted = np.zeros(len(state), dtype=float)
        else:
            standardized = np.clip((state - self.state_center) / self.state_scale, -6.0, 6.0)
            fitted = standardized @ self.beta
        regime = (fitted - self.fit_center) / self.fit_scale
        weights = 1.0 + self.regime_strength * np.clip(regime, -2.0, 2.0)
        main_values, _ = self._smooth(main * weights[:, None], self.main_span)
        diversifier_values, _ = self._smooth(diversifier, self.diversifier_span)
        result = pd.DataFrame(main_values + self.blend_weight * diversifier_values, index=features.index, columns=tickers)
        return result.sub(result.mean(axis=1), axis=0).fillna(0.0)


if __name__ == "__main__":
    result = walkforward_full(StatelessOrbitProbe, gauge_fix=True, verbose=True)
    print(f"net_sharpe={result['sharpe']:+.6f}")
    print(f"city_proxy={result['city_novelty_deg']:.2f}")
