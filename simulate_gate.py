import numpy as np
import pandas as pd
import xgboost as xgb
from data_loader import load_period
from alphanova_orbit_hack import OrbitHack

def _mse(pred, target):
    p = pred.values.reshape(-1)
    t = target.values.reshape(-1)
    ok = np.isfinite(p) & np.isfinite(t)
    return np.mean((p[ok] - t[ok])**2)

def _ic(pred, target):
    p = pred.values.reshape(-1)
    t = target.values.reshape(-1)
    ok = np.isfinite(p) & np.isfinite(t)
    return np.corrcoef(p[ok], t[ok])[0, 1]

def run_gate(w):
    data = load_period('031', 'data/train', 'small')
    f = data['features']
    t = data['target']
    
    n_train = len(f) - 800
    f_train = f.iloc[:n_train]
    t_train = t.iloc[:n_train]
    f_val = f.iloc[n_train:]
    t_val = t.iloc[n_train:]
    
    # Real
    model = OrbitHack()
    model.w = w
    model.train(f_train, t_train)
    pred_real = model.predict(f_val)
    
    ic_real = _ic(pred_real, t_val)
    mse_real = _mse(pred_real, t_val)
    
    # Permuted
    np.random.seed(42)
    ic_null = []
    mse_null = []
    
    for i in range(20):
        t_perm = t_train.copy()
        t_perm.values[:] = np.random.permutation(t_perm.values.reshape(-1)).reshape(t_perm.shape)
        
        m = OrbitHack()
        m.w = w
        m.train(f_train, t_perm)
        p = m.predict(f_val)
        
        ic_null.append(_ic(p, t_val))
        mse_null.append(_mse(p, t_val))
        print(f"Perm {i}: IC={ic_null[-1]:.6f}")
        
    ic_mean, ic_std = np.mean(ic_null), np.std(ic_null)
    z_ic = (ic_real - ic_mean) / ic_std if ic_std > 0 else 0
    
    mse_mean, mse_std = np.mean(mse_null), np.std(mse_null)
    z_mse = (mse_mean - mse_real) / mse_std if mse_std > 0 else 0
    
    print(f"\nWeight {w}:")
    print(f"IC : Real={ic_real:.6f}, NullMean={ic_mean:.6f}, NullStd={ic_std:.6f}, Z={z_ic:.2f}")
    print(f"MSE: Real={mse_real:.6f}, NullMean={mse_mean:.6f}, NullStd={mse_std:.6f}, Z={z_mse:.2f}")

if __name__ == "__main__":
    run_gate(0.01)
    run_gate(1.0)
