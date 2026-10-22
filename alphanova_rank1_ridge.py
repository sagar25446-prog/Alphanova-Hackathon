import numpy as np
import pandas as pd
from predictor import Predictor

class TopRankPredictor(Predictor):
    def __init__(self):
        super().__init__()
        self.alpha = 1000.0
        self.weights = None
        self.last_signal = None
        self.smooth_factor = 0.12

    def _engineer_features(self, df):
        if isinstance(df.index, pd.MultiIndex):
            mean = df.groupby(level=0).transform('mean')
            std = df.groupby(level=0).transform('std').replace(0, 1e-8)
            z = (df - mean) / std
        else:
            mean = df.mean(axis=1) if df.ndim > 1 else df.mean()
            std = df.std(axis=1) if df.ndim > 1 else df.std()
            std = std.replace(0, 1e-8)
            z = df.sub(mean, axis=0).div(std, axis=0) if df.ndim > 1 else (df - mean) / std

        z_cols = list(z.columns)
        features_list = [z]
        new_cols = list(z.columns)

        try:
            f1 = z['Feature.1']
            f2 = z['Feature.2']
            f3 = z['Feature.3']
            f4 = z['Feature.4']
            f5 = z['Feature.5']
            f6 = z['Feature.6']
            
            core = (f1 + f2 + f3 + f4) / 4.0
            negcore3 = -np.power(core, 3)
            primary = negcore3 + (core * f6)
            
            # Since core is a DataFrame with ticker columns, we can just append it
            # We don't really care about the column names as long as the dimensions match
            # But pd.concat might complain about duplicate columns, so we can just use values
            features_list.append(core)
            features_list.append(negcore3)
            features_list.append(primary)
        except KeyError:
            pass

        n_cols = len(z_cols)
        for i in range(n_cols):
            col_i = z[z_cols[i]]
            features_list.append(np.tanh(col_i))
            features_list.append(col_i * np.abs(col_i))
            for j in range(i + 1, n_cols):
                col_j = z[z_cols[j]]
                features_list.append(col_i * col_j)

        # Concatenate using numpy to avoid any MultiIndex or column name issues
        X_vals = np.hstack([f.values if hasattr(f, 'values') else f for f in features_list])
        X = pd.DataFrame(X_vals, index=z.index)
        X = X.fillna(0.0).replace([np.inf, -np.inf], 0.0)
        return X

    def train(self, features, target):
        X = self._engineer_features(features)
        
        if isinstance(target, (pd.DataFrame, pd.Series)):
            y = target.values.ravel()
        else:
            y = np.asarray(target).ravel()

        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        X_vals = X.values
        
        y = -y
        
        n_features = X_vals.shape[1]
        XtX = X_vals.T @ X_vals
        Xty = X_vals.T @ y
        
        np.fill_diagonal(XtX, XtX.diagonal() + self.alpha)
        
        self.weights = np.linalg.solve(XtX, Xty)

    def predict(self, features):
        X = self._engineer_features(features)
        raw_preds = X.values @ self.weights
        
        if isinstance(features.index, pd.MultiIndex):
            tickers = features.columns.get_level_values(1).unique()
        else:
            tickers = features.columns
            
        pred_series = pd.DataFrame(raw_preds.reshape(-1, len(tickers)), index=features.index.unique(level=0) if isinstance(features.index, pd.MultiIndex) else features.index, columns=tickers)

        if isinstance(pred_series.index, pd.MultiIndex):
            xs_mean = pred_series.groupby(level=0).transform('mean')
            demeaned = pred_series - xs_mean
        else:
            demeaned = pred_series.sub(pred_series.mean(axis=1), axis=0)

        if self.last_signal is None or len(self.last_signal) != len(demeaned):
            smoothed = demeaned
        else:
            smoothed = self.smooth_factor * demeaned + (1.0 - self.smooth_factor) * self.last_signal

        if isinstance(pred_series.index, pd.MultiIndex):
            final_pred = smoothed - smoothed.groupby(level=0).transform('mean')
        else:
            final_pred = smoothed.sub(smoothed.mean(axis=1), axis=0)

        self.last_signal = final_pred.copy()
        return final_pred
