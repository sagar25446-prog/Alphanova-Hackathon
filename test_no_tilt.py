import sys
sys.path.insert(0, ".")
from alphanova_f5_engineered import F5EngineeredPredictor
from walkforward import walkforward_full

class NoTiltPredictor(F5EngineeredPredictor):
    def predict(self, features):
        import pandas as pd
        import xgboost as xgb
        import numpy as np
        X, tickers, index = self._design(features)
        raw = self.model.predict(xgb.DMatrix(np.nan_to_num(X)))
        pred_matrix = pd.DataFrame(
            raw.reshape(-1, len(tickers)), index=index, columns=tickers,
        )
        pred_matrix = pred_matrix.ewm(span=self.span, min_periods=1).mean()
        # NO CAUSAL TILT REMOVAL
        pred_matrix = pred_matrix.sub(pred_matrix.mean(axis=1), axis=0)
        return pred_matrix.fillna(0.0)

print(walkforward_full(NoTiltPredictor, "data/train", "small", verbose=False))
