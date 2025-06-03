"""Unit tests for automl_mini.utils module."""

import numpy as np
import pandas as pd
import pytest

from automl_mini.utils import DataValidationError, detect_problem_type, validate_data


class TestValidateData:
    """Test cases for validate_data function."""

    def test_valid_dataframe_and_series(self):
        """Test with valid pandas DataFrame and Series."""
        X = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        y = pd.Series([0, 1, 0], name="target")

        X_validated, y_validated = validate_data(X, y)

        assert isinstance(X_validated, pd.DataFrame)
        assert isinstance(y_validated, pd.Series)
        assert len(X_validated) == len(y_validated) == 3
        assert list(X_validated.columns) == ["feature1", "feature2"]
        assert y_validated.name == "target"

    def test_numpy_arrays(self):
        """Test with numpy arrays - should convert to pandas."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])

        X_validated, y_validated = validate_data(X, y)

        assert isinstance(X_validated, pd.DataFrame)
        assert isinstance(y_validated, pd.Series)
        assert len(X_validated) == len(y_validated) == 3
        assert list(X_validated.columns) == ["feature_0", "feature_1"]
        assert y_validated.name == "target"

    def test_mixed_input_types(self):
        """Test with mixed input types."""
        X = np.array([[1, 2], [3, 4]])
        y = pd.Series([0, 1], name="labels")

        X_validated, y_validated = validate_data(X, y)

        assert isinstance(X_validated, pd.DataFrame)
        assert isinstance(y_validated, pd.Series)
        assert y_validated.name == "labels"

    def test_none_inputs(self):
        """Test with None inputs - should raise error."""
        with pytest.raises(DataValidationError, match="Input X and y cannot be None"):
            validate_data(None, pd.Series([1, 2, 3]))

        with pytest.raises(DataValidationError, match="Input X and y cannot be None"):
            validate_data(pd.DataFrame({"a": [1, 2]}), None)

    def test_empty_dataframe(self):
        """Test with empty DataFrame - should raise error."""
        X = pd.DataFrame()
        y = pd.Series([1, 2, 3])

        with pytest.raises(DataValidationError, match="Input X cannot be empty"):
            validate_data(X, y)

    def test_empty_series(self):
        """Test with empty Series - should raise error."""
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([], dtype=float)

        with pytest.raises(DataValidationError, match="Input y cannot be empty"):
            validate_data(X, y)

    def test_mismatched_lengths(self):
        """Test with mismatched sample lengths - should raise error."""
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([1, 2])

        with pytest.raises(
            DataValidationError, match="X and y must have the same number of samples"
        ):
            validate_data(X, y)

    def test_insufficient_samples(self):
        """Test with insufficient samples - should raise error."""
        X = pd.DataFrame({"a": [1]})
        y = pd.Series([1])

        with pytest.raises(
            DataValidationError, match="Need at least 2 samples for training"
        ):
            validate_data(X, y)

    def test_invalid_input_types(self):
        """Test with invalid input types - should raise error."""
        with pytest.raises(
            DataValidationError, match="X must be a pandas DataFrame or numpy array"
        ):
            validate_data("invalid", pd.Series([1, 2, 3]))

        with pytest.raises(
            DataValidationError, match="y must be a pandas Series or numpy array"
        ):
            validate_data(pd.DataFrame({"a": [1, 2, 3]}), "invalid")

    def test_index_reset(self):
        """Test that indices are reset properly."""
        X = pd.DataFrame({"a": [1, 2, 3]}, index=[10, 20, 30])
        y = pd.Series([1, 2, 3], index=[10, 20, 30])

        X_validated, y_validated = validate_data(X, y)

        assert list(X_validated.index) == [0, 1, 2]
        assert list(y_validated.index) == [0, 1, 2]


class TestDetectProblemType:
    """Test cases for detect_problem_type function."""

    def test_classification_integer_limited_unique(self):
        """Test classification detection with integer target and limited unique values."""
        y = pd.Series([0, 1, 2, 0, 1, 2], dtype="int64")
        result = detect_problem_type(y)
        assert result == "classification"

    def test_classification_integer_many_unique(self):
        """Test with integer target but many unique values - should be regression."""
        y = pd.Series(range(25), dtype="int64")  # 25 unique values
        result = detect_problem_type(y)
        assert result == "regression"

    def test_classification_string_target(self):
        """Test classification detection with string target."""
        y = pd.Series(["cat", "dog", "bird", "cat", "dog"])
        result = detect_problem_type(y)
        assert result == "classification"

    def test_regression_float_target(self):
        """Test regression detection with float target."""
        y = pd.Series([1.5, 2.3, 3.7, 4.1, 5.9])
        result = detect_problem_type(y)
        assert result == "regression"

    def test_classification_boundary_case(self):
        """Test boundary case with exactly 20 unique integer values."""
        y = pd.Series(list(range(20)), dtype="int64")
        result = detect_problem_type(y)
        assert result == "classification"

    def test_regression_boundary_case(self):
        """Test boundary case with 21 unique integer values."""
        y = pd.Series(list(range(21)), dtype="int64")
        result = detect_problem_type(y)
        assert result == "regression"

    def test_classification_boolean_target(self):
        """Test classification with boolean-like integer target."""
        y = pd.Series([0, 1, 0, 1, 0], dtype="int64")
        result = detect_problem_type(y)
        assert result == "classification"
