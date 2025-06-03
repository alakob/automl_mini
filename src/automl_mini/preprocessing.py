"""
Data preprocessing module for AutoML Simple.

This module follows SOLID principles:
- Single Responsibility: Each transformer handles one type of preprocessing
- Open/Closed: Easy to extend with new transformers
- Liskov Substitution: All transformers implement the same interface
- Interface Segregation: Clean, minimal transformer interface
- Dependency Inversion: Depends on abstractions, not concrete implementations
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


class FeatureType(Enum):
    """Enumeration of feature types for type safety."""

    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"


class BaseTransformer(ABC):
    """
    Abstract base class for all data transformers.

    This follows the Interface Segregation Principle by providing
    a minimal, focused interface that all transformers must implement.
    """

    def __init__(self):
        self.is_fitted_ = False

    @abstractmethod
    def fit(self, X: pd.DataFrame) -> "BaseTransformer":
        """Fit the transformer to the data."""
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        pass

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def _check_is_fitted(self) -> None:
        """Check if transformer is fitted."""
        if not self.is_fitted_:
            raise ValueError("Transformer must be fitted before transform")


class NumericalTransformer(BaseTransformer):
    """
    Transformer for numerical features.

    Handles missing values and scaling for numerical data.
    Follows Single Responsibility Principle.
    """

    def __init__(self):
        super().__init__()
        self.imputer = SimpleImputer(strategy="mean")
        self.scaler = StandardScaler()
        self.feature_names_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame) -> "NumericalTransformer":
        """Fit the numerical transformer."""
        if X.empty:
            raise ValueError("Cannot fit on empty DataFrame")

        # Store feature names
        self.feature_names_ = list(X.columns)

        # Fit imputer and scaler
        X_imputed = self.imputer.fit_transform(X)
        self.scaler.fit(X_imputed)

        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform numerical features."""
        self._check_is_fitted()

        if X.empty:
            return X

        # Apply imputation and scaling
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)

        # Return as DataFrame with original column names
        result = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        return result


class CategoricalTransformer(BaseTransformer):
    """
    Transformer for categorical features.

    Handles missing values and encoding for categorical data.
    Follows Single Responsibility Principle.
    """

    def __init__(self, max_categories: int = 10):
        super().__init__()
        self.max_categories = max_categories
        self.imputer = SimpleImputer(strategy="most_frequent")
        self.encoders: Dict[str, Any] = {}
        self.feature_names_: Optional[List[str]] = None
        self.output_columns_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame) -> "CategoricalTransformer":
        """Fit the categorical transformer."""
        if X.empty:
            raise ValueError("Cannot fit on empty DataFrame")

        self.feature_names_ = list(X.columns)
        self.output_columns_ = []

        # Fit imputer
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X), columns=X.columns, index=X.index
        )

        # Fit encoders for each column
        for col in X.columns:
            unique_values = X_imputed[col].nunique()

            if unique_values <= self.max_categories:
                # Use OneHotEncoder for low cardinality
                encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                encoder.fit(X_imputed[[col]])

                # Generate output column names
                if hasattr(encoder, "get_feature_names_out"):
                    feature_names = encoder.get_feature_names_out([col])
                else:
                    feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]

                self.output_columns_.extend(feature_names)

            else:
                # Use LabelEncoder for high cardinality
                encoder = LabelEncoder()
                encoder.fit(X_imputed[col])
                self.output_columns_.append(col)

            self.encoders[col] = encoder

        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical features."""
        self._check_is_fitted()

        if X.empty:
            return X

        # Apply imputation
        X_imputed = pd.DataFrame(
            self.imputer.transform(X), columns=X.columns, index=X.index
        )

        # Apply encoding
        encoded_dfs = []

        for col in X.columns:
            encoder = self.encoders[col]

            if isinstance(encoder, OneHotEncoder):
                # One-hot encoding
                encoded = encoder.transform(X_imputed[[col]])

                if hasattr(encoder, "get_feature_names_out"):
                    feature_names = encoder.get_feature_names_out([col])
                else:
                    feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]

                encoded_df = pd.DataFrame(encoded, columns=feature_names, index=X.index)

            else:
                # Label encoding
                encoded = encoder.transform(X_imputed[col])
                encoded_df = pd.DataFrame({col: encoded}, index=X.index)

            encoded_dfs.append(encoded_df)

        # Concatenate all encoded features
        if encoded_dfs:
            result = pd.concat(encoded_dfs, axis=1)
        else:
            result = pd.DataFrame(index=X.index)

        return result


class DataPreprocessor:
    """
    Main preprocessing coordinator that orchestrates different transformers.

    This follows the Open/Closed Principle - it's open for extension
    (can add new transformers) but closed for modification.
    Also follows Dependency Inversion - depends on transformer abstractions.
    """

    def __init__(self):
        self.transformers: Dict[FeatureType, BaseTransformer] = {}
        self.feature_types_: Optional[Dict[str, FeatureType]] = None
        self.is_fitted_ = False

    def fit(self, X: pd.DataFrame) -> "DataPreprocessor":
        """
        Fit the preprocessor to the data.

        Args:
            X: Input features

        Returns:
            Self for method chaining
        """
        if X.empty:
            raise ValueError("Cannot fit on empty DataFrame")

        # Detect feature types
        self.feature_types_ = self._detect_feature_types(X)

        # Group features by type
        feature_groups = self._group_features_by_type(X)

        # Fit transformers for each feature type
        for feature_type, columns in feature_groups.items():
            if not columns:
                continue

            if feature_type == FeatureType.NUMERICAL:
                transformer = NumericalTransformer()
            elif feature_type == FeatureType.CATEGORICAL:
                transformer = CategoricalTransformer()
            else:
                continue

            transformer.fit(X[columns])
            self.transformers[feature_type] = transformer

        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data.

        Args:
            X: Input features

        Returns:
            Transformed features
        """
        if not self.is_fitted_:
            raise ValueError("Preprocessor must be fitted before transform")

        if X.empty:
            return X

        # Group features by type
        feature_groups = self._group_features_by_type(X)

        # Transform each feature group
        transformed_dfs = []

        for feature_type, columns in feature_groups.items():
            if not columns or feature_type not in self.transformers:
                continue

            transformer = self.transformers[feature_type]
            transformed = transformer.transform(X[columns])
            transformed_dfs.append(transformed)

        # Concatenate all transformed features
        if transformed_dfs:
            result = pd.concat(transformed_dfs, axis=1)
        else:
            result = pd.DataFrame(index=X.index)

        return result

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def _detect_feature_types(self, X: pd.DataFrame) -> Dict[str, FeatureType]:
        """Detect feature types for each column."""
        feature_types = {}

        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                feature_types[col] = FeatureType.NUMERICAL
            else:
                feature_types[col] = FeatureType.CATEGORICAL

        return feature_types

    def _group_features_by_type(self, X: pd.DataFrame) -> Dict[FeatureType, List[str]]:
        """Group feature names by their detected types."""
        if self.feature_types_ is None:
            raise ValueError("Feature types not detected. Call fit first.")

        groups: Dict[FeatureType, List[str]] = {
            FeatureType.NUMERICAL: [],
            FeatureType.CATEGORICAL: [],
        }

        for col in X.columns:
            if col in self.feature_types_:
                feature_type = self.feature_types_[col]
                groups[feature_type].append(col)

        return groups
