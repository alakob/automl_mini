"""Integration tests for automl_mini.pipeline module."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from automl_mini import AutoMLPipeline, PipelineConfig, PipelineResult
from automl_mini.models import ProblemType
from automl_mini.pipeline import PipelineError


class TestPipelineConfig:
    """Test cases for PipelineConfig."""

    def test_default_initialization(self):
        """Test default configuration values."""
        config = PipelineConfig()

        assert config.cv_folds == 3
        assert config.test_size == 0.2
        assert config.random_state == 42
        assert config.problem_type is None
        assert config.verbose is False

    def test_custom_initialization(self):
        """Test custom configuration values."""
        config = PipelineConfig(
            cv_folds=5,
            test_size=0.3,
            random_state=123,
            problem_type=ProblemType.CLASSIFICATION,
            verbose=True,
        )

        assert config.cv_folds == 5
        assert config.test_size == 0.3
        assert config.random_state == 123
        assert config.problem_type == ProblemType.CLASSIFICATION
        assert config.verbose is True

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = PipelineConfig(cv_folds=5, problem_type=ProblemType.REGRESSION)

        config_dict = config.to_dict()

        assert config_dict["cv_folds"] == 5
        assert config_dict["problem_type"] == "regression"
        assert "test_size" in config_dict
        assert "random_state" in config_dict
        assert "verbose" in config_dict

    def test_from_dict(self):
        """Test creation from dictionary."""
        config_dict = {
            "cv_folds": 5,
            "test_size": 0.25,
            "problem_type": "classification",
            "verbose": True,
        }

        config = PipelineConfig.from_dict(config_dict)

        assert config.cv_folds == 5
        assert config.test_size == 0.25
        assert config.problem_type == ProblemType.CLASSIFICATION
        assert config.verbose is True

    def test_from_dict_with_none_problem_type(self):
        """Test creation from dictionary with None problem type."""
        config_dict = {"cv_folds": 3, "problem_type": None}

        config = PipelineConfig.from_dict(config_dict)

        assert config.problem_type is None


class TestPipelineResult:
    """Test cases for PipelineResult."""

    @pytest.fixture
    def sample_pipeline_result(self):
        """Create a sample pipeline result for testing."""
        from sklearn.ensemble import RandomForestClassifier

        from automl_mini.models import ModelResult

        model = RandomForestClassifier()
        model_results = [
            ModelResult(model, 0.85, np.array([0.8, 0.9, 0.85])),
            ModelResult(RandomForestClassifier(), 0.75, np.array([0.7, 0.8, 0.75])),
        ]

        return PipelineResult(
            best_model=model,
            best_score=0.85,
            problem_type=ProblemType.CLASSIFICATION,
            model_results=model_results,
            total_time=10.5,
            preprocessing_time=2.3,
            model_selection_time=8.2,
            original_features=["feature1", "feature2"],
            transformed_features=[
                "feature1_scaled",
                "feature2_onehot_A",
                "feature2_onehot_B",
            ],
        )

    def test_summary_generation(self, sample_pipeline_result):
        """Test summary string generation."""
        summary = sample_pipeline_result.summary()

        assert "AutoML Pipeline Results" in summary
        assert "Problem Type: classification" in summary
        assert "Best Model: RandomForestClassifier" in summary
        assert "Best Score: 0.8500" in summary
        assert "Total Time: 10.50 seconds" in summary
        assert "Model Comparison:" in summary
        assert "Features: 2 → 3" in summary


class TestAutoMLPipeline:
    """Test cases for AutoMLPipeline."""

    @pytest.fixture
    def classification_data(self):
        """Generate sample classification data."""
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_informative=3,
            n_redundant=1,
            n_classes=3,
            random_state=42,
        )

        # Convert to DataFrame with mixed feature types
        df = pd.DataFrame(X, columns=[f"num_feature_{i}" for i in range(5)])

        # Add categorical features using proper random generator
        rng = np.random.RandomState(42)
        df["cat_feature_1"] = rng.choice(["A", "B", "C"], size=len(df))
        df["cat_feature_2"] = rng.choice(["X", "Y"], size=len(df))

        target = pd.Series(y, name="target")

        return df, target

    @pytest.fixture
    def regression_data(self):
        """Generate sample regression data."""
        X, y = make_regression(n_samples=100, n_features=4, noise=0.1, random_state=42)

        # Convert to DataFrame with mixed feature types
        df = pd.DataFrame(X, columns=[f"num_feature_{i}" for i in range(4)])

        # Add categorical features using proper random generator
        rng = np.random.RandomState(42)
        df["cat_feature_1"] = rng.choice(["Type1", "Type2", "Type3"], size=len(df))

        target = pd.Series(y, name="target")

        return df, target

    def test_default_initialization(self):
        """Test default pipeline initialization."""
        pipeline = AutoMLPipeline()

        assert isinstance(pipeline.config, PipelineConfig)
        assert not pipeline.is_fitted_
        assert pipeline.pipeline_result_ is None
        assert pipeline.feature_names_ is None
        assert pipeline.target_name_ is None

    def test_initialization_with_config_dict(self):
        """Test initialization with configuration dictionary."""
        config_dict = {"cv_folds": 5, "verbose": True}
        pipeline = AutoMLPipeline(config=config_dict)

        assert pipeline.config.cv_folds == 5
        assert pipeline.config.verbose is True

    def test_initialization_with_config_object(self):
        """Test initialization with PipelineConfig object."""
        config = PipelineConfig(cv_folds=5, verbose=True)
        pipeline = AutoMLPipeline(config=config)

        assert pipeline.config.cv_folds == 5
        assert pipeline.config.verbose is True

    def test_invalid_config_type(self):
        """Test that invalid config type raises error."""
        with pytest.raises(PipelineError, match="Invalid config type"):
            AutoMLPipeline(config="invalid")

    def test_fit_classification_data(self, classification_data):
        """Test fitting with classification data."""
        X, y = classification_data
        pipeline = AutoMLPipeline()

        result = pipeline.fit(X, y)

        assert result is pipeline  # Should return self
        assert pipeline.is_fitted_
        assert pipeline.pipeline_result_ is not None
        assert pipeline.pipeline_result_.problem_type == ProblemType.CLASSIFICATION
        assert pipeline.pipeline_result_.best_score > 0
        assert len(pipeline.pipeline_result_.model_results) > 0
        assert pipeline.feature_names_ is not None
        assert pipeline.target_name_ == "target"

    def test_fit_regression_data(self, regression_data):
        """Test fitting with regression data."""
        X, y = regression_data
        pipeline = AutoMLPipeline()

        pipeline.fit(X, y)

        assert pipeline.is_fitted_
        assert pipeline.pipeline_result_.problem_type == ProblemType.REGRESSION
        assert pipeline.pipeline_result_.best_score > 0

    def test_fit_with_explicit_problem_type(self, classification_data):
        """Test fitting with explicitly specified problem type."""
        X, y = classification_data
        config = PipelineConfig(problem_type=ProblemType.CLASSIFICATION)
        pipeline = AutoMLPipeline(config=config)

        pipeline.fit(X, y)

        assert pipeline.pipeline_result_.problem_type == ProblemType.CLASSIFICATION

    def test_fit_with_verbose_output(self, classification_data, capsys):
        """Test fitting with verbose output."""
        X, y = classification_data
        config = PipelineConfig(verbose=True)
        pipeline = AutoMLPipeline(config=config)

        pipeline.fit(X, y)

        captured = capsys.readouterr()
        assert "Training pipeline on" in captured.out
        assert "Step 1: Preprocessing data" in captured.out
        assert "Step 2: Model selection" in captured.out

    def test_predict_after_fit(self, classification_data):
        """Test prediction after fitting."""
        X, y = classification_data
        pipeline = AutoMLPipeline()
        pipeline.fit(X, y)

        predictions = pipeline.predict(X)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X)

    def test_predict_proba_classification(self, classification_data):
        """Test probability prediction for classification."""
        X, y = classification_data
        pipeline = AutoMLPipeline()
        pipeline.fit(X, y)

        probabilities = pipeline.predict_proba(X)

        assert isinstance(probabilities, np.ndarray)
        assert probabilities.shape[0] == len(X)
        assert probabilities.shape[1] == len(np.unique(y))  # Number of classes
        # Check that probabilities sum to 1 for each sample
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_predict_proba_regression_error(self, regression_data):
        """Test that predict_proba raises error for regression."""
        X, y = regression_data
        pipeline = AutoMLPipeline()
        pipeline.fit(X, y)

        with pytest.raises(
            PipelineError, match="predict_proba is only available for classification"
        ):
            pipeline.predict_proba(X)

    def test_score_method(self, classification_data):
        """Test scoring method."""
        X, y = classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        pipeline = AutoMLPipeline()
        pipeline.fit(X_train, y_train)

        score = pipeline.score(X_test, y_test)

        assert isinstance(score, float)
        assert 0 <= score <= 1  # F1 score should be between 0 and 1

    def test_predict_before_fit(self, classification_data):
        """Test that prediction before fit raises error."""
        X, y = classification_data
        pipeline = AutoMLPipeline()

        with pytest.raises(PipelineError, match="Pipeline must be fitted before use"):
            pipeline.predict(X)

    def test_predict_proba_before_fit(self, classification_data):
        """Test that predict_proba before fit raises error."""
        X, y = classification_data
        pipeline = AutoMLPipeline()

        with pytest.raises(PipelineError, match="Pipeline must be fitted before use"):
            pipeline.predict_proba(X)

    def test_get_results_before_fit(self):
        """Test that get_results before fit raises error."""
        pipeline = AutoMLPipeline()

        with pytest.raises(PipelineError, match="Pipeline must be fitted before use"):
            pipeline.get_results()

    def test_get_results_after_fit(self, classification_data):
        """Test getting results after fitting."""
        X, y = classification_data
        pipeline = AutoMLPipeline()
        pipeline.fit(X, y)

        results = pipeline.get_results()

        assert isinstance(results, PipelineResult)
        assert results.best_score > 0
        assert results.total_time > 0
        assert len(results.model_results) > 0

    def test_get_feature_importance(self, classification_data):
        """Test getting feature importance."""
        X, y = classification_data
        pipeline = AutoMLPipeline()
        pipeline.fit(X, y)

        importance = pipeline.get_feature_importance()

        # Should return feature importance if available
        if importance is not None:
            assert isinstance(importance, dict)
            assert len(importance) > 0
            # All importance values should be numeric
            assert all(isinstance(v, (int, float)) for v in importance.values())

    def test_prediction_with_numpy_array(self, classification_data):
        """Test prediction with numpy array input."""
        X, y = classification_data
        pipeline = AutoMLPipeline()
        pipeline.fit(X, y)

        # Convert to numpy for prediction
        X_np = X.values
        predictions = pipeline.predict(X_np)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X)

    def test_pipeline_repr(self, classification_data):
        """Test pipeline string representation."""
        X, y = classification_data
        pipeline = AutoMLPipeline()

        # Before fit
        repr_before = repr(pipeline)
        assert "fitted=False" in repr_before

        # After fit
        pipeline.fit(X, y)
        repr_after = repr(pipeline)
        assert "fitted=True" in repr_after
        assert "best_model=" in repr_after
        assert "score=" in repr_after

    def test_end_to_end_workflow_classification(self, classification_data):
        """Test complete end-to-end workflow for classification."""
        X, y = classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Create and configure pipeline
        config = PipelineConfig(cv_folds=3, verbose=False)
        pipeline = AutoMLPipeline(config=config)

        # Fit pipeline
        pipeline.fit(X_train, y_train)

        # Make predictions
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)

        # Evaluate
        score = pipeline.score(X_test, y_test)

        # Get results
        results = pipeline.get_results()

        # Assertions
        assert pipeline.is_fitted_
        assert len(predictions) == len(X_test)
        assert probabilities.shape[0] == len(X_test)
        assert score > 0
        assert results.problem_type == ProblemType.CLASSIFICATION
        assert results.best_score > 0
        assert results.total_time > 0

        # Check that summary works
        summary = results.summary()
        assert isinstance(summary, str)
        assert "AutoML Pipeline Results" in summary

    def test_end_to_end_workflow_regression(self, regression_data):
        """Test complete end-to-end workflow for regression."""
        X, y = regression_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Create and configure pipeline
        pipeline = AutoMLPipeline()

        # Fit pipeline
        pipeline.fit(X_train, y_train)

        # Make predictions
        predictions = pipeline.predict(X_test)

        # Evaluate
        score = pipeline.score(X_test, y_test)

        # Get results
        results = pipeline.get_results()

        # Assertions
        assert pipeline.is_fitted_
        assert len(predictions) == len(X_test)
        assert isinstance(score, float)
        assert results.problem_type == ProblemType.REGRESSION
        assert results.best_score is not None  # R² can be negative
        assert results.total_time > 0

    def test_error_handling_invalid_data(self):
        """Test error handling with invalid data."""
        pipeline = AutoMLPipeline()

        # Test with None data
        with pytest.raises(PipelineError, match="Pipeline fitting failed"):
            pipeline.fit(None, None)

    def test_error_handling_empty_data(self):
        """Test error handling with empty data."""
        pipeline = AutoMLPipeline()

        empty_X = pd.DataFrame()
        empty_y = pd.Series([], dtype=float)

        with pytest.raises(PipelineError, match="Pipeline fitting failed"):
            pipeline.fit(empty_X, empty_y)
