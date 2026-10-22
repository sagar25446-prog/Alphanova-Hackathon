"""Base predictor interface for the Bi-weekly competition, Season 1 (Walk-Forward).

All submissions must inherit from the Predictor class and implement
the train() and predict() methods. Do not modify this file.
"""

import warnings
from abc import ABC, abstractmethod

import pandas as pd

warnings.filterwarnings("ignore")


class Predictor(ABC):
    """Base class for all predictors in the walk-forward competition.

    All submissions must inherit from this class and implement the required methods.
    Your entire solution must be contained within your subclass — do not define
    helper functions outside of it.

    The predictor will be evaluated via a temporal split on the latest
    period's data (min(800, 20%) rows held out for validation):
        1. train() is called on the training portion
        2. predict() is called on the validation portion
        3. Predictions are scored against validation returns (Sharpe ratio)

    Submission format:
        Submit a single .py file (or .ipynb notebook) containing your Predictor
        subclass. Imports at the top of the file are fine. The following packages
        are pre-installed: numpy, pandas, scikit-learn, xgboost, lightgbm, pyarrow.
        For any additional dependencies, declare them via PEP 723 at the top of
        your file::

            # /// script
            # dependencies = ["catboost>=1.2"]
            # ///

        All logic must live inside your class — do not define helper functions or
        classes outside of it.

    Important notes:
        - Each period's data is independently obfuscated. Ticker identities are NOT
          consistent across periods (ticker.1 in period 001 != ticker.1 in period 002).
        - Your model must learn cross-sectional patterns, not ticker-specific ones.
        - Features have a MultiIndex: (feature_name, ticker) at level 0 and 1.
        - Predictions must be cross-sectionally de-meaned at each timestamp
          (i.e. sum across tickers = 0).
    """

    @abstractmethod
    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """Train the predictor on historical data.

        This will be called with concatenated data from multiple periods.
        Each period is independently obfuscated.

        Args:
            features: MultiIndex DataFrame (feature_name, ticker) with shape
                      (n_timestamps, n_features * n_tickers). There are 6
                      features (Feature.1 through Feature.6). Their
                      identities are not disclosed.
            target: DataFrame (n_timestamps, n_tickers) with cross-sectionally
                    de-meaned forward-looking values.
        """

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for a single period.

        Args:
            features: MultiIndex DataFrame (feature_name, ticker), same format
                      as in train(). This is one period of features.

        Returns:
            DataFrame (n_timestamps, n_tickers) with cross-sectionally de-meaned
            predictions. Must satisfy:
              - sum across tickers = 0 at each timestamp
        """
