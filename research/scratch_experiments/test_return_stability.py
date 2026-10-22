import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
import numpy as np
import pandas as pd
from data_loader import load_period

for p in ['001', '010', '020', '030', '031']:
    d = load_period(p, 'data/train', 'small')
    r = d['returns']
    for i in range(1, 7):
        f = d['features'][f'Feature.{i}']
        ic = np.nanmean(r.corrwith(f.shift(1), axis=1))
        print(f"Period {p} F{i} Return IC: {ic:.4f}")
    print("-" * 30)
