"""Offline ablation: remove the target-fitted regime multiplier."""

import numpy as np

from alphanova_prism_five_tilt import PrismFiveTiltPredictor
from walkforward import walkforward_full


class PrismFiveTiltNoGate(PrismFiveTiltPredictor):
    def __init__(self):
        super().__init__()
        self.regime_strength = 0.0


if __name__ == "__main__":
    result = walkforward_full(PrismFiveTiltNoGate, gauge_fix=True, verbose=True)
    print(f"net_sharpe={result['sharpe']:+.6f}")
    print(f"city_proxy={result.get('city_novelty_deg', float('nan')):.2f}")
