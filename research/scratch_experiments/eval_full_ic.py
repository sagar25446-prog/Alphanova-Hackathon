import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
import numpy as np
import pandas as pd
from test_primary import PrimaryPredictor
from data_loader import list_periods, load_period

def eval_full():
    periods = sorted(list_periods("data/train"))
    predictor = PrimaryPredictor()
    
    all_ics = []
    
    for p in periods:
        d = load_period(p, "data/train", "small")
        pred = predictor.predict(d["features"])
        target = d["target"].reindex(pred.index)
        
        ic = np.nanmean(target.corrwith(pred, axis=1))
        all_ics.append(ic)
        
    print(f"Full Walkforward Avg IC: {np.mean(all_ics):.4f}")

if __name__ == "__main__":
    eval_full()
