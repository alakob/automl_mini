"""Unit tests for automl_mini.preprocessing module."""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from automl_mini.preprocessing import (
    BaseTransformer,
    CategoricalTransformer,
    DataPreprocessor,
    FeatureType,
    NumericalTransformer,
)


class TestBaseTransformer:
    """Test cases for BaseTransformer abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseTransformer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseTransformer()


class TestNumericalTransformer:
    """Test cases for NumericalTransformer."""

    @pytest.fixture
    def sample_numerical_data(self):
        """Sample numerical data for testing."""
        return pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )

    @pytest.fixture
    def numerical_data_with_missing(self):
        """Numerical data with missing values."""
        return pd.DataFrame(
            {
                "feature1": [1.0, np.nan, 3.0, 4.0, 5.0],
                "feature2": [10.0, 20.0, np.nan, 40.0, 50.0],
            }
        )

    def test_initialization(self):
        """Test NumericalTransformer initialization."""
        transformer = NumericalTransformer()
        assert not transformer.is_fitted_
        assert transformer.feature_names_ is None

    def test_fit_method(self, sample_numerical_data):
        """Test the fit method."""
        transformer = NumericalTransformer()
        result = transformer.fit(sample_numerical_data)

        # Should return self for method chaining
        assert result is transformer
        assert transformer.is_fitted_
        assert transformer.feature_names_ == ["feature1", "feature2"]

    def test_transform_method(self, sample_numerical_data):
        """Test the transform method."""
        transformer = NumericalTransformer()
        transformer.fit(sample_numerical_data)

        transformed = transformer.transform(sample_numerical_data)

        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == sample_numerical_data.shape
        assert list(transformed.columns) == list(sample_numerical_data.columns)

        # Check that data is standardized (mean ~0, std ~1)
        assert abs(transformed["feature1"].mean()) < 1e-10
        assert abs(transformed["feature1"].std(ddof=0) - 1.0) < 1e-1

    def test_fit_transform_method(self, sample_numerical_data):
        """Test the fit_transform method."""
        transformer = NumericalTransformer()
        transformed = transformer.fit_transform(sample_numerical_data)

        assert transformer.is_fitted_
        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == sample_numerical_data.shape

    def test_transform_before_fit(self, sample_numerical_data):
        """Test that transform raises error before fit."""
        transformer = NumericalTransformer()

        with pytest.raises(
            ValueError, match="Transformer must be fitted before transform"
        ):
            transformer.transform(sample_numerical_data)

    def test_fit_empty_dataframe(self):
        """Test fit with empty DataFrame."""
        transformer = NumericalTransformer()
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="Cannot fit on empty DataFrame"):
            transformer.fit(empty_df)

    def test_transform_empty_dataframe(self, sample_numerical_data):
        """Test transform with empty DataFrame."""
        transformer = NumericalTransformer()
        transformer.fit(sample_numerical_data)

        empty_df = pd.DataFrame()
        result = transformer.transform(empty_df)

        assert result.empty

    def test_missing_value_handling(self, numerical_data_with_missing):
        """Test handling of missing values."""
        transformer = NumericalTransformer()
        transformed = transformer.fit_transform(numerical_data_with_missing)

        # Should have no missing values after transformation
        assert not transformed.isnull().any().any()
        assert transformed.shape == numerical_data_with_missing.shape


class TestCategoricalTransformer:
    """Test cases for CategoricalTransformer."""

    @pytest.fixture
    def sample_categorical_data(self):
        """Sample categorical data for testing."""
        return pd.DataFrame(
            {
                "category1": ["A", "B", "C", "A", "B"],
                "category2": ["X", "Y", "X", "Y", "X"],
            }
        )

    @pytest.fixture
    def high_cardinality_data(self):
        """High cardinality categorical data."""
        categories = [f"cat_{i}" for i in range(15)]  # 15 categories
        # Ensure we use all categories by creating a longer array
        # and guaranteeing all categories appear
        rng = np.random.RandomState(42)

        # First, include all categories at least once
        data = list(categories)  # 15 samples with all categories
        # Then add random choices to reach 50 total samples
        data.extend(rng.choice(categories, 35))

        # Shuffle the final data
        rng.shuffle(data)

        return pd.DataFrame({"high_card": data})

    @pytest.fixture
    def categorical_data_with_missing(self):
        """Categorical data with missing values."""
        return pd.DataFrame(
            {
                "category1": ["A", np.nan, "C", "A", "B"],
                "category2": ["X", "Y", np.nan, "Y", "X"],
            }
        )

    def test_initialization(self):
        """Test CategoricalTransformer initialization."""
        transformer = CategoricalTransformer()
        assert not transformer.is_fitted_
        assert transformer.max_categories == 10
        assert transformer.feature_names_ is None

    def test_initialization_with_custom_max_categories(self):
        """Test initialization with custom max_categories."""
        transformer = CategoricalTransformer(max_categories=5)
        assert transformer.max_categories == 5

    def test_fit_low_cardinality(self, sample_categorical_data):
        """Test fitting with low cardinality data (should use OneHotEncoder)."""
        transformer = CategoricalTransformer()
        result = transformer.fit(sample_categorical_data)

        assert result is transformer
        assert transformer.is_fitted_
        assert transformer.feature_names_ == ["category1", "category2"]

        # Should use OneHotEncoder for low cardinality
        for col in sample_categorical_data.columns:
            assert isinstance(transformer.encoders[col], OneHotEncoder)

    def test_fit_high_cardinality(self, high_cardinality_data):
        """Test fitting with high cardinality data (should use LabelEncoder)."""
        transformer = CategoricalTransformer(max_categories=10)
        transformer.fit(high_cardinality_data)

        # Should use LabelEncoder for high cardinality
        assert isinstance(transformer.encoders["high_card"], LabelEncoder)

    def test_transform_onehot_encoding(self, sample_categorical_data):
        """Test transform with OneHot encoding."""
        transformer = CategoricalTransformer()
        transformed = transformer.fit_transform(sample_categorical_data)

        assert isinstance(transformed, pd.DataFrame)
        # OneHot encoding should increase number of columns
        assert transformed.shape[1] > sample_categorical_data.shape[1]
        assert transformed.shape[0] == sample_categorical_data.shape[0]

    def test_transform_label_encoding(self, high_cardinality_data):
        """Test transform with Label encoding."""
        transformer = CategoricalTransformer(max_categories=10)
        transformed = transformer.fit_transform(high_cardinality_data)

        assert isinstance(transformed, pd.DataFrame)
        # Label encoding should keep same number of columns
        assert transformed.shape == high_cardinality_data.shape
        # Should contain only numeric values
        assert pd.api.types.is_numeric_dtype(transformed["high_card"])

    def test_missing_value_handling(self, categorical_data_with_missing):
        """Test handling of missing values."""
        transformer = CategoricalTransformer()
        transformed = transformer.fit_transform(categorical_data_with_missing)

        # Should have no missing values after transformation
        assert not transformed.isnull().any().any()


class TestDataPreprocessor:
    """Test cases for DataPreprocessor."""

    @pytest.fixture
    def mixed_data(self):
        """Mixed numerical and categorical data."""
        return pd.DataFrame(
            {
                "num1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "num2": [10.0, 20.0, 30.0, 40.0, 50.0],
                "cat1": ["A", "B", "C", "A", "B"],
                "cat2": ["X", "Y", "X", "Y", "X"],
            }
        )

    @pytest.fixture
    def only_numerical_data(self):
        """Only numerical data."""
        return pd.DataFrame(
            {"num1": [1.0, 2.0, 3.0, 4.0, 5.0], "num2": [10.0, 20.0, 30.0, 40.0, 50.0]}
        )

    @pytest.fixture
    def only_categorical_data(self):
        """Only categorical data."""
        return pd.DataFrame(
            {"cat1": ["A", "B", "C", "A", "B"], "cat2": ["X", "Y", "X", "Y", "X"]}
        )

    def test_initialization(self):
        """Test DataPreprocessor initialization."""
        preprocessor = DataPreprocessor()
        assert not preprocessor.is_fitted_
        assert preprocessor.feature_types_ is None
        assert len(preprocessor.transformers) == 0

    def test_feature_type_detection(self, mixed_data):
        """Test automatic feature type detection."""
        preprocessor = DataPreprocessor()
        feature_types = preprocessor._detect_feature_types(mixed_data)

        assert feature_types["num1"] == FeatureType.NUMERICAL
        assert feature_types["num2"] == FeatureType.NUMERICAL
        assert feature_types["cat1"] == FeatureType.CATEGORICAL
        assert feature_types["cat2"] == FeatureType.CATEGORICAL

    def test_fit_mixed_data(self, mixed_data):
        """Test fitting with mixed data types."""
        preprocessor = DataPreprocessor()
        result = preprocessor.fit(mixed_data)

        assert result is preprocessor
        assert preprocessor.is_fitted_
        assert preprocessor.feature_types_ is not None
        assert FeatureType.NUMERICAL in preprocessor.transformers
        assert FeatureType.CATEGORICAL in preprocessor.transformers

    def test_fit_only_numerical(self, only_numerical_data):
        """Test fitting with only numerical data."""
        preprocessor = DataPreprocessor()
        preprocessor.fit(only_numerical_data)

        assert FeatureType.NUMERICAL in preprocessor.transformers
        assert FeatureType.CATEGORICAL not in preprocessor.transformers

    def test_fit_only_categorical(self, only_categorical_data):
        """Test fitting with only categorical data."""
        preprocessor = DataPreprocessor()
        preprocessor.fit(only_categorical_data)

        assert FeatureType.CATEGORICAL in preprocessor.transformers
        assert FeatureType.NUMERICAL not in preprocessor.transformers

    def test_transform_mixed_data(self, mixed_data):
        """Test transforming mixed data."""
        preprocessor = DataPreprocessor()
        transformed = preprocessor.fit_transform(mixed_data)

        assert isinstance(transformed, pd.DataFrame)
        assert len(transformed) == len(mixed_data)
        # Should have more columns due to categorical encoding
        assert len(transformed.columns) >= len(mixed_data.columns)

    def test_transform_before_fit(self, mixed_data):
        """Test that transform raises error before fit."""
        preprocessor = DataPreprocessor()

        with pytest.raises(
            ValueError, match="Preprocessor must be fitted before transform"
        ):
            preprocessor.transform(mixed_data)

    def test_fit_empty_dataframe(self):
        """Test fit with empty DataFrame."""
        preprocessor = DataPreprocessor()
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="Cannot fit on empty DataFrame"):
            preprocessor.fit(empty_df)

    def test_transform_empty_dataframe(self, mixed_data):
        """Test transform with empty DataFrame."""
        preprocessor = DataPreprocessor()
        preprocessor.fit(mixed_data)

        empty_df = pd.DataFrame()
        result = preprocessor.transform(empty_df)

        assert result.empty

    def test_group_features_by_type(self, mixed_data):
        """Test grouping features by type."""
        preprocessor = DataPreprocessor()
        preprocessor.feature_types_ = preprocessor._detect_feature_types(mixed_data)

        groups = preprocessor._group_features_by_type(mixed_data)

        assert "num1" in groups[FeatureType.NUMERICAL]
        assert "num2" in groups[FeatureType.NUMERICAL]
        assert "cat1" in groups[FeatureType.CATEGORICAL]
        assert "cat2" in groups[FeatureType.CATEGORICAL]

    def test_group_features_before_fit(self, mixed_data):
        """Test that grouping features raises error before fit."""
        preprocessor = DataPreprocessor()

        with pytest.raises(
            ValueError, match="Feature types not detected. Call fit first."
        ):
            preprocessor._group_features_by_type(mixed_data)


class TestFeatureType:
    """Test cases for FeatureType enum."""

    def test_enum_values(self):
        """Test FeatureType enum values."""
        assert FeatureType.NUMERICAL.value == "numerical"
        assert FeatureType.CATEGORICAL.value == "categorical"

    def test_enum_comparison(self):
        """Test enum comparison."""
        assert FeatureType.NUMERICAL == FeatureType.NUMERICAL
        assert FeatureType.NUMERICAL != FeatureType.CATEGORICAL
