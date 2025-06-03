"""
Utility functions for data validation and basic operations.

This module follows the Single Responsibility Principle by focusing solely
on data validation and format conversion utilities.
"""

from typing import Tuple, Union

import numpy as np
import pandas as pd


class DataValidationError(Exception):
    """Exception raised for data validation errors."""

    pass


def validate_data(
    X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Validate and convert input data to pandas DataFrame and Series.

    Args:
        X: Feature data
        y: Target data

    Returns:
        Tuple of (DataFrame, Series) with validated data

    Raises:
        DataValidationError: If data is invalid
    """
    # Check for None inputs
    if X is None or y is None:
        raise DataValidationError("Input X and y cannot be None")

    # Convert to pandas if needed
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    elif not isinstance(X, pd.DataFrame):
        raise DataValidationError("X must be a pandas DataFrame or numpy array")

    if isinstance(y, np.ndarray):
        y = pd.Series(y, name="target")
    elif not isinstance(y, pd.Series):
        raise DataValidationError("y must be a pandas Series or numpy array")

    # Check for empty data
    if X.empty:
        raise DataValidationError("Input X cannot be empty")

    if len(y) == 0:
        raise DataValidationError("Input y cannot be empty")

    # Check shape compatibility
    if len(X) != len(y):
        raise DataValidationError(
            f"X and y must have the same number of samples. "
            f"X has {len(X)}, y has {len(y)}"
        )

    # Check for minimum samples
    if len(X) < 2:
        raise DataValidationError("Need at least 2 samples for training")

    # Reset indices to ensure alignment
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    return X, y


def detect_problem_type(y: pd.Series) -> str:
    """
    Automatically detect if the problem is classification or regression.

    Args:
        y: Target series

    Returns:
        'classification' or 'regression'
    """
    # If target has non-numeric data, it's classification
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"

    # If target is numeric, check number of unique values
    unique_values = y.nunique()

    # If fewer than 20 unique values and target is integer-like, treat as classification
    if unique_values <= 20 and pd.api.types.is_integer_dtype(y):
        return "classification"

    # Otherwise, treat as regression
    return "regression"
