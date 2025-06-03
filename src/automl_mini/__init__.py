"""
AutoML Mini - A streamlined automated machine learning library

This package provides core automated machine learning capabilities including:
- Data preprocessing
- Model selection
- Basic evaluation
- Simple pipeline management

Example:
    >>> from automl_mini import AutoMLPipeline
    >>> pipeline = AutoMLPipeline()
    >>> pipeline.fit(X_train, y_train)
    >>> predictions = pipeline.predict(X_test)
"""

from .models import ModelSelector, ProblemType
from .pipeline import AutoMLPipeline, PipelineConfig, PipelineResult
from .preprocessing import DataPreprocessor, FeatureType
from .utils import DataValidationError, validate_data

__version__ = "1.0.0"
__author__ = "AutoML Mini Team"

__all__ = [
    # Core pipeline
    "AutoMLPipeline",
    "PipelineConfig",
    "PipelineResult",
    # Preprocessing
    "DataPreprocessor",
    "FeatureType",
    # Model selection
    "ModelSelector",
    "ProblemType",
    # Utilities
    "validate_data",
    "DataValidationError",
]
