"""Data loading utilities for the Bi-weekly competition, Season 1 (Walk-Forward).

Provides functions to discover, load, and concatenate training data.
Do not modify this file.
"""

import re
from pathlib import Path

import pandas as pd


def list_periods(data_dir: str = "data/train", universe: str = "small") -> list[str]:
    """List all available training periods, sorted chronologically.

    Args:
        data_dir: Directory containing parquet files.
        universe: Universe type suffix ('small', 'top20vol', or 'top50').

    Returns:
        Sorted list of period keys, e.g. ['001', '002', ...].
    """
    data_path = Path(data_dir)
    pattern = re.compile(
        rf"^(\d{{3}})\.{re.escape(universe)}\.features\.parquet$"
    )

    periods = []
    for f in data_path.iterdir():
        m = pattern.match(f.name)
        if m:
            periods.append(m.group(1))

    return sorted(periods)




def load_period(
    period_key: str,
    data_dir: str = "data/train",
    universe: str = "small",
) -> dict[str, pd.DataFrame]:
    """Load a single period's training data.

    Args:
        period_key: Period identifier, e.g. '001'.
        data_dir: Directory containing parquet files.
        universe: Universe type suffix.

    Returns:
        Dictionary with keys 'features', 'returns', 'target'.
    """
    data_path = Path(data_dir)
    prefix = f"{period_key}.{universe}"

    features = pd.read_parquet(data_path / f"{prefix}.features.parquet")
    returns = pd.read_parquet(data_path / f"{prefix}.returns.parquet")
    target = pd.read_parquet(data_path / f"{prefix}.target.parquet")

    return {"features": features, "returns": returns, "target": target}


def load_latest_training_period(
    period_keys: list[str],
    data_dir: str = "data/train",
    universe: str = "small",
) -> dict[str, pd.DataFrame]:
    """Load just the most recent training period from a list.

    Period files are *cumulative*: period N's parquet contains every
    training row from competition start through the end of month N-1.
    So `load_periods([001, ..., 064])` would concatenate duplicates —
    the earliest row 64 times, the second-earliest 63 times, and so
    on — biasing the training set heavily toward early history. The
    correct pattern is to load only `period_keys[-1]`.

    Raises ValueError if `period_keys` is empty.
    """
    if not period_keys:
        raise ValueError(
            "load_latest_training_period: period_keys is empty. The caller "
            "should filter by test-month before calling."
        )
    return load_period(period_keys[-1], data_dir, universe)


def load_periods(
    period_keys: list[str],
    data_dir: str = "data/train",
    universe: str = "small",
) -> dict[str, pd.DataFrame]:
    """Load and concatenate data from multiple periods.

    Each period's data is loaded and concatenated along the time axis.
    The integer index is reset to be contiguous across all periods.

    Args:
        period_keys: List of period identifiers to load.
        data_dir: Directory containing parquet files.
        universe: Universe type suffix.

    Returns:
        Dictionary with keys 'features', 'returns', 'target',
        each a concatenated DataFrame across all requested periods.
    """
    all_features = []
    all_returns = []
    all_targets = []

    for key in period_keys:
        data = load_period(key, data_dir, universe)
        all_features.append(data["features"])
        all_returns.append(data["returns"])
        all_targets.append(data["target"])

    features = pd.concat(all_features, axis=0, ignore_index=True)
    returns = pd.concat(all_returns, axis=0, ignore_index=True)
    target = pd.concat(all_targets, axis=0, ignore_index=True)

    return {"features": features, "returns": returns, "target": target}




def split_walkforward(
    periods: list[str],
    n_validate: int = 12,
) -> tuple[list[str], list[str]]:
    """Split periods into training and validation sets for walk-forward.

    The last n_validate periods become the validation window.
    The walk-forward engine will expand the training window period by period
    within this validation window.

    Args:
        periods: Chronologically sorted list of period keys.
        n_validate: Number of periods to reserve for validation.

    Returns:
        Tuple of (train_periods, validate_periods).
    """
    if n_validate >= len(periods):
        raise ValueError(
            f"n_validate ({n_validate}) must be less than total periods ({len(periods)})"
        )

    cutoff = len(periods) - n_validate
    return periods[:cutoff], periods[cutoff:]
