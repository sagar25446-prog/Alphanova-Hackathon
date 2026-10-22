"""Sweep to find the maximum-concentration, minimum-turnover variant.

Target: beat clearcutmeal at +0.0177 by maximizing concentration
(reducing turnover) while preserving IC. Heavy smoothing is the key.
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from walkforward import walkforward_full
from alphanova_gate_pass_f5blend import GatePassingF5Blend

DATA_DIR = "data/train"
UNIVERSE = "small"

configs = [
    # (primary_span, sleeve_span, weight, label)
    (120, 480, 2.0, "BASE (TaroKogawa-match)"),
    (160, 480, 2.0, "p160_s480_w2"),
    (200, 480, 2.0, "p200_s480_w2"),
    (240, 480, 2.0, "p240_s480_w2"),
    (120, 600, 2.0, "p120_s600_w2"),
    (120, 720, 2.0, "p120_s720_w2"),
    (120, 480, 3.0, "p120_s480_w3"),
    (120, 480, 4.0, "p120_s480_w4"),
    (160, 600, 3.0, "p160_s600_w3"),
    (200, 720, 3.0, "p200_s720_w3"),
    (240, 720, 4.0, "p240_s720_w4"),
    (360, 720, 4.0, "p360_s720_w4"),
    (120, 480, 0.0, "PURE_PRISM (w=0)"),
    (360, 360, 0.0, "PURE_PRISM_SLOW"),
]

print(f"{'Config':<28s} {'Sharpe':>10s}")
print("-" * 42)

for pspan, sspan, w, label in configs:
    class V(GatePassingF5Blend):
        def __init__(self):
            super().__init__()
            self.primary_span = pspan
            self.sleeve_span = sspan
            self.weight = w
    try:
        r = walkforward_full(V, DATA_DIR, UNIVERSE, verbose=False)
        print(f"{label:<28s} {r.get('sharpe',0):>+10.6f}")
    except Exception as e:
        print(f"{label:<28s} ERROR: {e}")
