"""
Main AutoML pipeline module.

This module orchestrates the complete machine learning workflow by coordinating
preprocessing and model selection components. It follows SOLID principles by
depending on abstractions and providing a clean, unified interface.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from .models import ModelResult, ModelSelector, ProblemType
from .preprocessing import DataPreprocessor
from .utils import validate_data


class PipelineError(Exception):
    """Exception raised for pipeline-related errors."""

    pass


@dataclass
class PipelineConfig:
    """Configuration for the AutoML pipeline."""

    # Cross-validation settings
    cv_folds: int = 3

    # Test split settings
    test_size: float = 0.2

    # Random state for reproducibility
    random_state: int = 42

    # Problem type (auto-detected if None)
    problem_type: Optional[ProblemType] = None

    # Verbosity
    verbose: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "cv_folds": self.cv_folds,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "problem_type": self.problem_type.value if self.problem_type else None,
            "verbose": self.verbose,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PipelineConfig":
        """Create configuration from dictionary."""
        # Handle ProblemType enum
        if "problem_type" in config_dict and config_dict["problem_type"]:
            config_dict = config_dict.copy()
            config_dict["problem_type"] = ProblemType(config_dict["problem_type"])

        return cls(**config_dict)


@dataclass
class PipelineResult:
    """Results from the AutoML pipeline execution."""

    # Core results
    best_model: BaseEstimator
    best_score: float
    problem_type: ProblemType

    # Model evaluation results
    model_results: List[ModelResult]

    # Timing information
    total_time: float
    preprocessing_time: float
    model_selection_time: float

    # Feature information
    original_features: List[str] = field(default_factory=list)
    transformed_features: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Generate a summary of the pipeline results."""
        lines = [
            "AutoML Pipeline Results",
            "=" * 50,
            f"Problem Type: {self.problem_type.value}",
            f"Best Model: {type(self.best_model).__name__}",
            f"Best Score: {self.best_score:.4f}",
            f"Total Time: {self.total_time:.2f} seconds",
            "",
            "Model Comparison:",
        ]

        for i, result in enumerate(self.model_results, 1):
            lines.append(f"  {i}. {result.model_name}: {result.score:.4f}")

        lines.extend(
            [
                "",
                f"Features: {len(self.original_features)} → {len(self.transformed_features)}",
                f"Preprocessing Time: {self.preprocessing_time:.2f}s",
                f"Model Selection Time: {self.model_selection_time:.2f}s",
            ]
        )

        return "\n".join(lines)


class AutoMLPipeline:
    """
    Main AutoML pipeline that orchestrates preprocessing and model selection.

    This class follows the Dependency Inversion Principle by depending on
    abstractions (DataPreprocessor, ModelSelector) rather than concrete
    implementations. It provides a clean, unified interface for the entire
    AutoML workflow.

    Example:
        >>> pipeline = AutoMLPipeline()
        >>> pipeline.fit(X_train, y_train)
        >>> predictions = pipeline.predict(X_test)
    """

    def __init__(self, config: Optional[Union[PipelineConfig, Dict[str, Any]]] = None):
        """
        Initialize the AutoML pipeline.

        Args:
            config: Pipeline configuration (PipelineConfig object or dict)
        """
        # Handle configuration
        if config is None:
            self.config = PipelineConfig()
        elif isinstance(config, dict):
            self.config = PipelineConfig.from_dict(config)
        elif isinstance(config, PipelineConfig):
            self.config = config
        else:
            raise PipelineError(f"Invalid config type: {type(config)}")

        # Initialize components
        self.preprocessor = DataPreprocessor()
        self.model_selector = ModelSelector(
            cv_folds=self.config.cv_folds, random_state=self.config.random_state
        )

        # State tracking
        self.is_fitted_ = False
        self.pipeline_result_: Optional[PipelineResult] = None

        # Store original data info
        self.feature_names_: Optional[List[str]] = None
        self.target_name_: Optional[str] = None

    def fit(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]
    ) -> "AutoMLPipeline":
        """
        Fit the AutoML pipeline to the training data.

        Args:
            X: Feature data
            y: Target data

        Returns:
            Self for method chaining

        Raises:
            PipelineError: If fitting fails
        """
        start_time = time.time()

        try:
            # Validate and convert input data
            X_df, y_series = validate_data(X, y)

            # Store metadata
            self.feature_names_ = list(X_df.columns)
            self.target_name_ = y_series.name or "target"

            if self.config.verbose:
                print(
                    f"Training pipeline on {len(X_df)} samples with {len(X_df.columns)} features"
                )

            # Step 1: Preprocessing
            preprocessing_start = time.time()

            if self.config.verbose:
                print("Step 1: Preprocessing data...")

            X_transformed = self.preprocessor.fit_transform(X_df)

            preprocessing_time = time.time() - preprocessing_start

            if self.config.verbose:
                print(f"  - Preprocessing completed in {preprocessing_time:.2f}s")
                print(
                    f"  - Features: {len(X_df.columns)} → {len(X_transformed.columns)}"
                )

            # Step 2: Model Selection
            model_selection_start = time.time()

            if self.config.verbose:
                print("Step 2: Model selection and evaluation...")

            self.model_selector.fit(
                X_transformed, y_series, problem_type=self.config.problem_type
            )

            model_selection_time = time.time() - model_selection_start

            if self.config.verbose:
                print(f"  - Model selection completed in {model_selection_time:.2f}s")
                best_model_name = type(self.model_selector.get_best_model()).__name__
                best_score = self.model_selector.get_best_score()
                print(f"  - Best model: {best_model_name} (score: {best_score:.4f})")

            # Calculate total time
            total_time = time.time() - start_time

            # Create pipeline result
            self.pipeline_result_ = PipelineResult(
                best_model=self.model_selector.get_best_model(),
                best_score=self.model_selector.get_best_score(),
                problem_type=self.model_selector.problem_type_,
                model_results=self.model_selector.get_model_results(),
                total_time=total_time,
                preprocessing_time=preprocessing_time,
                model_selection_time=model_selection_time,
                original_features=list(X_df.columns),
                transformed_features=list(X_transformed.columns),
            )

            self.is_fitted_ = True

            if self.config.verbose:
                print(f"Pipeline training completed in {total_time:.2f}s")

            return self

        except Exception as e:
            raise PipelineError(f"Pipeline fitting failed: {str(e)}") from e

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions using the fitted pipeline.

        Args:
            X: Feature data for prediction

        Returns:
            Predictions array

        Raises:
            PipelineError: If pipeline is not fitted or prediction fails
        """
        self._check_is_fitted()

        try:
            # Convert to DataFrame if needed
            if isinstance(X, np.ndarray):
                if self.feature_names_ is None:
                    raise PipelineError(
                        "Cannot predict with numpy array - no feature names stored"
                    )
                X_df = pd.DataFrame(X, columns=self.feature_names_)
            else:
                X_df = X.copy()

            # Apply preprocessing
            X_transformed = self.preprocessor.transform(X_df)

            # Make predictions
            predictions = self.model_selector.predict(X_transformed)

            return predictions

        except Exception as e:
            raise PipelineError(f"Prediction failed: {str(e)}") from e

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make probability predictions (classification only).

        Args:
            X: Feature data for prediction

        Returns:
            Probability predictions array

        Raises:
            PipelineError: If pipeline is not fitted, not classification, or prediction fails
        """
        self._check_is_fitted()

        if self.pipeline_result_.problem_type != ProblemType.CLASSIFICATION:
            raise PipelineError(
                "predict_proba is only available for classification problems"
            )

        try:
            # Convert to DataFrame if needed
            if isinstance(X, np.ndarray):
                if self.feature_names_ is None:
                    raise PipelineError(
                        "Cannot predict with numpy array - no feature names stored"
                    )
                X_df = pd.DataFrame(X, columns=self.feature_names_)
            else:
                X_df = X.copy()

            # Apply preprocessing
            X_transformed = self.preprocessor.transform(X_df)

            # Make probability predictions
            probabilities = self.model_selector.predict_proba(X_transformed)

            return probabilities

        except Exception as e:
            raise PipelineError(f"Probability prediction failed: {str(e)}") from e

    def score(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]
    ) -> float:
        """
        Score the pipeline on test data.

        Args:
            X: Test features
            y: Test targets

        Returns:
            Score on test data
        """
        self._check_is_fitted()

        # Validate test data
        X_df, y_series = validate_data(X, y)

        # Make predictions
        predictions = self.predict(X_df)

        # Calculate score based on problem type
        if self.pipeline_result_.problem_type == ProblemType.CLASSIFICATION:
            from sklearn.metrics import f1_score

            return f1_score(y_series, predictions, average="weighted")
        else:
            from sklearn.metrics import r2_score

            return r2_score(y_series, predictions)

    def get_results(self) -> PipelineResult:
        """
        Get the pipeline results.

        Returns:
            PipelineResult object with detailed results
        """
        self._check_is_fitted()
        return self.pipeline_result_

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance from the best model (if available).

        For tree-based models, uses feature_importances_.
        For linear models, uses absolute values of coefficients.

        Returns:
            Dictionary mapping feature names to importance scores, or None if not available
        """
        self._check_is_fitted()

        best_model = self.pipeline_result_.best_model
        feature_names = self.pipeline_result_.transformed_features

        # Tree-based models (RandomForest, GradientBoosting, etc.)
        if hasattr(best_model, "feature_importances_"):
            importance_scores = best_model.feature_importances_
            return {
                name: float(score)
                for name, score in zip(feature_names, importance_scores)
            }

        # Linear models (LogisticRegression, LinearRegression, etc.)
        elif hasattr(best_model, "coef_"):
            import numpy as np

            coef = best_model.coef_

            # Handle multiclass classification (coef_ is 2D)
            if coef.ndim > 1:
                # For multiclass, take the mean absolute coefficient across classes
                importance_scores = np.mean(np.abs(coef), axis=0)
            else:
                # For binary classification or regression
                importance_scores = np.abs(coef)

            # Normalize to make it comparable to feature_importances_
            if importance_scores.sum() > 0:
                importance_scores = importance_scores / importance_scores.sum()

            return {
                name: float(score)
                for name, score in zip(feature_names, importance_scores)
            }

        # Models without interpretable feature importance
        return None

    def format_feature_importance(self, top_n: int = None) -> Optional[str]:
        """
        Get a formatted string representation of feature importance.

        Args:
            top_n: Number of top features to display (None for all)

        Returns:
            Formatted string of feature importance, or None if not available
        """
        importance = self.get_feature_importance()

        if importance is None:
            return None

        # Sort by importance (descending)
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        # Limit to top_n if specified
        if top_n is not None:
            sorted_importance = sorted_importance[:top_n]

        lines = ["Feature Importance:"]
        for i, (feature, score) in enumerate(sorted_importance, 1):
            lines.append(f"  {i}. {feature}: {score:.3f}")

        return "\n".join(lines)

    def _check_is_fitted(self) -> None:
        """Check if the pipeline is fitted."""
        if not self.is_fitted_:
            raise PipelineError("Pipeline must be fitted before use")

    def __repr__(self) -> str:
        """String representation of the pipeline."""
        if self.is_fitted_:
            best_model_name = type(self.pipeline_result_.best_model).__name__
            score = self.pipeline_result_.best_score
            return f"AutoMLPipeline(fitted=True, best_model={best_model_name}, score={score:.4f})"
        else:
            return "AutoMLPipeline(fitted=False)"
