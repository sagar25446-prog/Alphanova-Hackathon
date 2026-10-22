"""City computation tools for scientists.

Allows scientists to compute gauge-fixed vectors and cities from their
predictions, then compare against the published city database.

Gauge matrices are precomputed and shipped with each training period.
They encode the coordinate system needed to compare signals across
different obfuscations.

Usage:
    from city_tools import compute_gauge_fixed, compute_city, compute_ic
    from city_tools import compare_to_cities

    # Load shipped gauge matrices for your period
    import numpy as np
    gauge = np.load("data/train/gauge/075.small.gauge.npy")  # (n_val, 19, 20)

    # Your validation predictions (n_val, 20), cross-sectionally de-meaned
    # Columns must be in order: ticker.1, ticker.2, ..., ticker.20
    gf = compute_gauge_fixed(predictions.values, gauge)

    # Compute metrics
    ic = compute_ic(gf)
    city = compute_city(gf)

    # Compare against published cities
    results = compare_to_cities(city)
"""

import numpy as np
import pandas as pd


def compute_gauge_fixed(
    predictions: np.ndarray,
    gauge_matrices: np.ndarray,
) -> np.ndarray:
    """Compute gauge-fixed vectors using shipped gauge matrices.

    Args:
        predictions: shape (T, N) — cross-sectionally de-meaned predictions.
            Columns must be in ticker order (ticker.1, ticker.2, ..., ticker.N).
        gauge_matrices: shape (T, D, N) — shipped gauge matrices, where D = N-1.
            Loaded from data/train/gauge/{period}.{universe}.gauge.npy

    Returns:
        gauge_fixed: ndarray (T, D) — gauge-fixed unit vectors.
    """
    T = predictions.shape[0]
    if gauge_matrices.shape[0] != T:
        raise ValueError(
            f"Shape mismatch: {T} prediction rows vs "
            f"{gauge_matrices.shape[0]} gauge matrices"
        )

    # Vectorized: einsum applies all per-row rotations in one call,
    # then we normalize per-row. Faster than the Python-level for-loop
    # by ~10-50x for typical T.
    raw = np.einsum("tij,tj->ti", gauge_matrices, predictions)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    gf = np.where(norms > 1e-12, raw / np.where(norms > 0, norms, 1.0), 0.0)
    return gf


def compute_city(gauge_fixed: np.ndarray) -> np.ndarray:
    """Compute city = normalized mean of gauge-fixed vectors.

    The city is a single point on S^{D-1} summarizing where the signal
    "lives" on average.

    Args:
        gauge_fixed: shape (T, D) — gauge-fixed unit vectors.

    Returns:
        Unit vector in R^D (the city).

    Raises:
        ValueError: If the mean vector is zero (degenerate signal).
    """
    v = gauge_fixed.mean(axis=0)
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        raise ValueError("Mean vector is zero — city is undefined (degenerate signal).")
    return v / norm


def compute_ic(gauge_fixed: np.ndarray) -> dict:
    """Compute information coefficient statistics from gauge-fixed vectors.

    After gauge-fixing, gf[t, 0] = cross-sectional correlation between
    the signal and the target at time t.

    Args:
        gauge_fixed: shape (T, D) — gauge-fixed unit vectors.

    Returns:
        Dict with 'ic_mean', 'ic_std', 'concentration'.
    """
    ic_series = gauge_fixed[:, 0]
    mean_vec = gauge_fixed.mean(axis=0)

    return {
        "ic_mean": float(np.mean(ic_series)),
        "ic_std": float(np.std(ic_series)),
        "concentration": float(np.linalg.norm(mean_vec)),
    }


def angular_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Angular distance between two unit vectors (radians).

    Args:
        a, b: Unit vectors of the same dimension.

    Returns:
        Angle in radians.
    """
    return float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))


def compare_to_cities(
    my_city: np.ndarray,
    city_db_path: str = "signal_db/signal_cities_ambient_coords.parquet",
) -> pd.DataFrame:
    """Compare your city against the published city database.

    Args:
        my_city: Unit vector in R^D (from compute_city).
        city_db_path: Path to the published city database parquet file.

    Returns:
        DataFrame with columns: signal_id, angular_distance_rad,
        angular_distance_deg, sorted by distance (nearest first).
    """
    db = pd.read_parquet(city_db_path)

    # Extract city coordinates (city_dim_0, city_dim_1, ..., city_dim_18)
    dim_cols = sorted(
        [c for c in db.columns if c.startswith("city_dim_")],
        key=lambda c: int(c.split("_")[-1]),
    )
    if not dim_cols:
        raise ValueError(f"No dim.* columns found in {city_db_path}")

    cities = db[dim_cols].values  # (K, D)

    # Normalize
    norms = np.linalg.norm(cities, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    cities = cities / norms

    # Compute angular distances
    dots = cities @ my_city
    np.clip(dots, -1.0, 1.0, out=dots)
    distances_rad = np.arccos(dots)

    # Build result
    id_col = "predictor_id" if "predictor_id" in db.columns else db.index.name or "index"
    ids = db[id_col].values if id_col in db.columns else db.index.values

    result = pd.DataFrame({
        "signal_id": ids,
        "angular_distance_rad": distances_rad,
        "angular_distance_deg": np.degrees(distances_rad),
    })

    return result.sort_values("angular_distance_rad").reset_index(drop=True)
