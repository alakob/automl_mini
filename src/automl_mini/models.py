"""
Model selection module for AutoML Simple.

This module follows SOLID principles:
- Single Responsibility: Each class has one clear purpose
- Open/Closed: Easy to add new models without modifying existing code
- Liskov Substitution: All models can be used interchangeably
- Interface Segregation: Clean, focused interfaces
- Dependency Inversion: Depends on sklearn abstractions
"""

import warnings
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score


class ProblemType(Enum):
    """Enumeration of problem types for type safety."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class ModelResult:
    """Container for model evaluation results."""

    def __init__(
        self, model: BaseEstimator, score: float, cv_scores: Optional[np.ndarray] = None
    ):
        self.model = model
        self.score = score
        self.cv_scores = cv_scores
        self.model_name = type(model).__name__

    def __repr__(self) -> str:
        cv_info = ""
        if self.cv_scores is not None:
            cv_mean = np.mean(self.cv_scores)
            cv_std = np.std(self.cv_scores)
            cv_info = f" (CV: {cv_mean:.4f} ± {cv_std:.4f})"

        return f"ModelResult({self.model_name}, score={self.score:.4f}{cv_info})"


class BaseModelFactory(ABC):
    """
    Abstract factory for creating models.

    This follows the Open/Closed Principle - we can add new model types
    without modifying existing code.
    """

    @abstractmethod
    def create_models(self) -> Dict[str, BaseEstimator]:
        """Create and return a dictionary of models."""
        pass

    @abstractmethod
    def get_scoring_metric(self) -> str:
        """Get the appropriate scoring metric for this problem type."""
        pass


class ClassificationModelFactory(BaseModelFactory):
    """Factory for creating classification models."""

    def create_models(self) -> Dict[str, BaseEstimator]:
        """Create classification models with sensible defaults."""
        return {
            "random_forest": RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            "logistic_regression": LogisticRegression(
                random_state=42, max_iter=1000, n_jobs=-1
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=100, random_state=42
            ),
        }

    def get_scoring_metric(self) -> str:
        """Return appropriate scoring metric for classification."""
        return "f1_weighted"


class RegressionModelFactory(BaseModelFactory):
    """Factory for creating regression models."""

    def create_models(self) -> Dict[str, BaseEstimator]:
        """Create regression models with sensible defaults."""
        return {
            "random_forest": RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            "linear_regression": LinearRegression(n_jobs=-1),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=100, random_state=42
            ),
        }

    def get_scoring_metric(self) -> str:
        """Return appropriate scoring metric for regression."""
        return "r2"


class ModelEvaluator:
    """
    Evaluates models using cross-validation.

    Follows Single Responsibility Principle by focusing only on evaluation.
    """

    def __init__(self, cv_folds: int = 3, random_state: int = 42):
        self.cv_folds = cv_folds
        self.random_state = random_state

    def evaluate_model(
        self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series, scoring: str
    ) -> ModelResult:
        """
        Evaluate a single model using cross-validation.

        Args:
            model: The model to evaluate
            X: Feature data
            y: Target data
            scoring: Scoring metric to use

        Returns:
            ModelResult with evaluation scores
        """
        try:
            # Suppress warnings during evaluation
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Perform cross-validation
                cv_scores = cross_val_score(
                    model, X, y, cv=self.cv_folds, scoring=scoring, n_jobs=-1
                )

                # Calculate mean score
                mean_score = np.mean(cv_scores)

                return ModelResult(model=model, score=mean_score, cv_scores=cv_scores)

        except Exception as e:
            # Return a result with score 0 if evaluation fails
            print(f"Warning: Failed to evaluate {type(model).__name__}: {e}")
            return ModelResult(model=model, score=0.0, cv_scores=None)

    def evaluate_models(
        self,
        models: Dict[str, BaseEstimator],
        X: pd.DataFrame,
        y: pd.Series,
        scoring: str,
    ) -> List[ModelResult]:
        """
        Evaluate multiple models.

        Args:
            models: Dictionary of model name to model instance
            X: Feature data
            y: Target data
            scoring: Scoring metric to use

        Returns:
            List of ModelResult objects sorted by score (descending)
        """
        results = []

        for name, model in models.items():
            result = self.evaluate_model(model, X, y, scoring)
            results.append(result)

        # Sort by score (descending)
        results.sort(key=lambda x: x.score, reverse=True)

        return results


class ModelSelector:
    """
    Main model selection class that orchestrates the entire process.

    This follows the Dependency Inversion Principle by depending on
    abstractions (BaseModelFactory) rather than concrete implementations.
    """

    def __init__(self, cv_folds: int = 3, random_state: int = 42):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.evaluator = ModelEvaluator(cv_folds, random_state)
        self.problem_type_: Optional[ProblemType] = None
        self.best_model_: Optional[BaseEstimator] = None
        self.model_results_: Optional[List[ModelResult]] = None

    def fit(
        self, X: pd.DataFrame, y: pd.Series, problem_type: Optional[ProblemType] = None
    ) -> "ModelSelector":
        """
        Fit the model selector to find the best model.

        Args:
            X: Feature data
            y: Target data
            problem_type: Problem type (auto-detected if None)

        Returns:
            Self for method chaining
        """
        # Detect or validate problem type
        if problem_type is None:
            self.problem_type_ = self._detect_problem_type(y)
        else:
            self.problem_type_ = problem_type

        # Get appropriate model factory
        factory = self._get_model_factory(self.problem_type_)

        # Create models
        models = factory.create_models()

        # Get scoring metric
        scoring = factory.get_scoring_metric()

        # Evaluate all models
        self.model_results_ = self.evaluator.evaluate_models(models, X, y, scoring)

        # Select best model
        if self.model_results_:
            best_result = self.model_results_[0]
            self.best_model_ = best_result.model

            # Fit the best model on full data
            self.best_model_.fit(X, y)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the best model."""
        if self.best_model_ is None:
            raise ValueError("ModelSelector must be fitted before predict")

        return self.best_model_.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions (classification only)."""
        if self.best_model_ is None:
            raise ValueError("ModelSelector must be fitted before predict_proba")

        if self.problem_type_ != ProblemType.CLASSIFICATION:
            raise ValueError("predict_proba only available for classification")

        if not hasattr(self.best_model_, "predict_proba"):
            raise ValueError(
                f"{type(self.best_model_).__name__} does not support predict_proba"
            )

        return self.best_model_.predict_proba(X)

    def get_best_model(self) -> BaseEstimator:
        """Get the best model."""
        if self.best_model_ is None:
            raise ValueError("ModelSelector must be fitted before getting best model")

        return self.best_model_

    def get_model_results(self) -> List[ModelResult]:
        """Get results for all evaluated models."""
        if self.model_results_ is None:
            raise ValueError("ModelSelector must be fitted before getting results")

        return self.model_results_

    def get_best_score(self) -> float:
        """Get the best model's score."""
        if self.model_results_ is None:
            raise ValueError("ModelSelector must be fitted before getting best score")

        return self.model_results_[0].score

    def _detect_problem_type(self, y: pd.Series) -> ProblemType:
        """Detect problem type from target variable."""
        if pd.api.types.is_numeric_dtype(y):
            # Check if all values are integers and limited unique values
            unique_values = y.nunique()
            if unique_values <= 20 and y.dtype in ["int64", "int32", "int16", "int8"]:
                return ProblemType.CLASSIFICATION
            else:
                return ProblemType.REGRESSION
        else:
            return ProblemType.CLASSIFICATION

    def _get_model_factory(self, problem_type: ProblemType) -> BaseModelFactory:
        """Get the appropriate model factory for the problem type."""
        if problem_type == ProblemType.CLASSIFICATION:
            return ClassificationModelFactory()
        elif problem_type == ProblemType.REGRESSION:
            return RegressionModelFactory()
        else:
            raise ValueError(f"Unsupported problem type: {problem_type}")
