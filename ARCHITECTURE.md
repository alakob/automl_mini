# AutoML Mini Architecture

This document provides a detailed technical overview of AutoML Mini's architecture, design patterns, and implementation of SOLID principles based on the **actual implementation**.

## 🏗️ Architecture Overview

AutoML Mini follows a **layered architecture** with clear separation of concerns:

```
┌─────────────────────────────────────┐
│           User Interface            │
│        (AutoMLPipeline)            │
├─────────────────────────────────────┤
│         Orchestration Layer        │
│       (Pipeline Management)        │
├─────────────────────────────────────┤
│        Processing Layer            │
│    (Preprocessing + Models)        │
├─────────────────────────────────────┤
│         Utility Layer              │
│    (Validation + Common Utils)     │
└─────────────────────────────────────┘
```

## 🔧 Core Components & Data Flow

### Component Structure
```
src/automl_mini/
├── __init__.py           # Public Interface
├── pipeline.py           # Main orchestration (AutoMLPipeline)
├── preprocessing.py      # Data preprocessing transformers
├── models.py            # Model selection and evaluation
└── utils.py             # Utilities and validation
```

### Data Flow Architecture
```
Input Data (X, y)
       ↓
┌─────────────────┐
│ DataPreprocessor │ ← Coordinates transformation
│ ┌─────────────┐ │
│ │ Numerical   │ │ ← Handles numeric features
│ │ Transformer │ │
│ └─────────────┘ │
│ ┌─────────────┐ │
│ │ Categorical │ │ ← Handles categorical features
│ │ Transformer │ │
│ └─────────────┘ │
└─────────────────┘
       ↓
┌─────────────────┐
│  ModelSelector  │ ← Evaluates models via factory
│ ┌─────────────┐ │
│ │ Model       │ │ ← Creates appropriate models
│ │ Factory     │ │
│ └─────────────┘ │
└─────────────────┘
       ↓
  Trained Pipeline
```

### Component Dependencies
```
AutoMLPipeline
    ├── PipelineConfig (Configuration)
    ├── DataPreprocessor (Preprocessing Service)
    │   ├── NumericalTransformer (Concrete Strategy)
    │   └── CategoricalTransformer (Concrete Strategy)
    ├── ModelSelector (Model Selection Service)
    │   ├── ClassificationModelFactory (Concrete Factory)
    │   ├── RegressionModelFactory (Concrete Factory)
    │   └── ModelEvaluator (Evaluation Service)
    └── ValidationUtils (Data Validation)
```

## 🎯 SOLID Principles Implementation

### 1. Single Responsibility Principle (SRP)

Each class has a single, well-defined responsibility:

```python
class DataPreprocessor:
    """ONLY responsible for coordinating data preprocessing."""

    def __init__(self):
        self.transformers = {}
        self.feature_types_ = None
        self.is_fitted_ = False

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Coordinates preprocessing workflow - single responsibility."""
        return self.fit(X).transform(X)

class ModelSelector:
    """ONLY responsible for model selection and evaluation."""

    def __init__(self, cv_folds: int = 3, random_state: int = 42):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.evaluator = ModelEvaluator(cv_folds, random_state)

    def fit(self, X: pd.DataFrame, y: pd.Series, problem_type=None):
        """ONLY handles model selection logic."""
        # Model selection implementation
        return self

class NumericalTransformer:
    """ONLY handles numerical feature transformation."""

    def fit(self, X: pd.DataFrame):
        """ONLY numerical feature fitting logic."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """ONLY numerical feature transformation."""
        # Imputation and scaling logic
        return transformed_data
```


### 2. Open/Closed Principle (OCP)

The system is **open for extension, closed for modification** through abstract base classes:

```python
from abc import ABC, abstractmethod

class BaseTransformer(ABC):
    """Abstract base class - extensible without modification."""

    def __init__(self):
        self.is_fitted_ = False

    @abstractmethod
    def fit(self, X: pd.DataFrame) -> "BaseTransformer":
        """Subclasses implement specific fitting logic."""
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Subclasses implement specific transformation logic."""
        pass

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Template method - reusable for all transformers."""
        return self.fit(X).transform(X)

# Easy to add new transformers without modifying existing code
class NumericalTransformer(BaseTransformer):
    """Extends BaseTransformer - no existing code modified."""

    def fit(self, X: pd.DataFrame) -> "NumericalTransformer":
        # Numerical-specific implementation
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Numerical transformation implementation
        return X  # Simplified for example

class CategoricalTransformer(BaseTransformer):
    """Another extension - system remains closed to modification."""

    def fit(self, X: pd.DataFrame) -> "CategoricalTransformer":
        # Categorical-specific implementation
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Categorical transformation implementation
        return X  # Simplified for example
```

### 3. Liskov Substitution Principle (LSP)

All transformer implementations can be substituted for the base class:

```python
def process_data(transformer: BaseTransformer, data: pd.DataFrame) -> pd.DataFrame:
    """Works with ANY BaseTransformer implementation."""
    return transformer.fit_transform(data)

# All implementations work identically - LSP satisfied
transformers = [
    NumericalTransformer(),      # Concrete implementation
    CategoricalTransformer(),    # Another concrete implementation
]

for transformer in transformers:
    result = process_data(transformer, data)  # LSP maintained
```

### 4. Interface Segregation Principle (ISP)

Interfaces are focused and clients depend only on methods they use:

```python
class BaseModelFactory(ABC):
    """Focused interface for model creation only."""

    @abstractmethod
    def create_models(self) -> Dict[str, BaseEstimator]:
        """Only model creation responsibility."""
        pass

    @abstractmethod
    def get_scoring_metric(self) -> str:
        """Only scoring metric responsibility."""
        pass
    # No unrelated methods forced on implementations

class ModelEvaluator:
    """Separate class for evaluation concerns - ISP compliance."""

    def evaluate_model(self, model, X, y, scoring) -> ModelResult:
        """Only evaluation-related functionality."""
        # Cross-validation evaluation logic
        pass
```

### 5. Dependency Inversion Principle (DIP)

High-level modules depend on abstractions through constructor injection:

```python
class AutoMLPipeline:
    """Depends on abstractions, not concrete implementations."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        # Depend on abstractions through constructor injection
        self.preprocessor = DataPreprocessor()  # Injected dependency
        self.model_selector = ModelSelector(   # Injected dependency
            cv_folds=self.config.cv_folds,
            random_state=self.config.random_state
        )

    def fit(self, X, y):
        """High-level workflow depends on abstractions."""
        # Use injected dependencies without knowing concrete types
        X_transformed = self.preprocessor.fit_transform(X)
        self.model_selector.fit(X_transformed, y)
        return self
```

## 🏛️ Design Patterns Implementation

### 1. Factory Pattern

**Model Creation Strategy:**
```python
from enum import Enum
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

class ProblemType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"

class ClassificationModelFactory(BaseModelFactory):
    """Factory creates appropriate models for classification."""

    def create_models(self) -> Dict[str, BaseEstimator]:
        return {
            'random_forest': RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            'logistic_regression': LogisticRegression(
                random_state=42, max_iter=1000, n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100, random_state=42
            )
        }

    def get_scoring_metric(self) -> str:
        return 'f1_weighted'

class RegressionModelFactory(BaseModelFactory):
    """Factory creates appropriate models for regression."""

    def create_models(self) -> Dict[str, BaseEstimator]:
        return {
            'random_forest': RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            'linear_regression': LinearRegression(n_jobs=-1),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100, random_state=42
            )
        }

    def get_scoring_metric(self) -> str:
        return 'r2'
```

### 2. Strategy Pattern

**Problem Type Strategy Selection:**
```python
class ModelSelector:
    def _get_model_factory(self, problem_type: ProblemType) -> BaseModelFactory:
        """Strategy pattern for different problem types."""
        if problem_type == ProblemType.CLASSIFICATION:
            return ClassificationModelFactory()
        elif problem_type == ProblemType.REGRESSION:
            return RegressionModelFactory()
        else:
            raise ValueError(f"Unsupported problem type: {problem_type}")
```

### 3. Template Method Pattern

**Transformer Template:**
```python
class BaseTransformer(ABC):
    """Template method pattern defining transformation workflow."""

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Template method defining the algorithm structure."""
        return self.fit(X).transform(X)

    @abstractmethod
    def fit(self, X: pd.DataFrame) -> "BaseTransformer":
        """Subclasses implement specific fitting logic."""
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Subclasses implement specific transformation logic."""
        pass

    def _check_is_fitted(self) -> None:
        """Common validation used by all transformers."""
        if not self.is_fitted_:
            raise ValueError("Transformer must be fitted before transform")
```

## 🚀 Configuration and Extensibility

### Current Configuration Structure
```python
@dataclass
class PipelineConfig:
    """Actual configuration structure in the codebase."""
    cv_folds: int = 3
    test_size: float = 0.2
    random_state: int = 42
    problem_type: Optional[ProblemType] = None
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
        if "problem_type" in config_dict and config_dict["problem_type"]:
            config_dict = config_dict.copy()
            config_dict["problem_type"] = ProblemType(config_dict["problem_type"])
        return cls(**config_dict)
```

### Extensibility Points

**Adding New Transformers:**
1. Inherit from `BaseTransformer`
2. Implement `fit()` and `transform()` methods
3. Add to `DataPreprocessor` detection logic

**Adding New Models:**
1. Create new factory class inheriting from `BaseModelFactory`
2. Implement `create_models()` and `get_scoring_metric()`
3. Update strategy mapping in `ModelSelector._get_model_factory()`

**Adding New Problem Types:**
1. Extend `ProblemType` enum
2. Create corresponding factory implementation
3. Update strategy mapping

## 📊 Performance Features

### Actual Performance Implementation
The library implements several concrete performance optimizations:

- **Parallel Cross-Validation**: Uses `n_jobs=-1` in sklearn's `cross_val_score`
- **Parallel Model Training**: RandomForest and LogisticRegression use `n_jobs=-1`
- **Efficient Data Structures**: Built on NumPy/Pandas optimized operations
- **Memory Management**: Proper DataFrame index handling and column management

### Current Performance Code
```python
class ModelEvaluator:
    def evaluate_model(self, model, X, y, scoring) -> ModelResult:
        """Cross-validation with parallel processing where supported."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Parallel cross-validation - actual implementation
            cv_scores = cross_val_score(
                model, X, y, cv=self.cv_folds, scoring=scoring, n_jobs=-1
            )

            return ModelResult(
                model=model,
                score=np.mean(cv_scores),
                cv_scores=cv_scores
            )

# Model factories with parallel processing where available
class ClassificationModelFactory(BaseModelFactory):
    def create_models(self) -> Dict[str, BaseEstimator]:
        return {
            'random_forest': RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1  # Parallel
            ),
            'logistic_regression': LogisticRegression(
                random_state=42, max_iter=1000, n_jobs=-1     # Parallel
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100, random_state=42             # Sequential
            )
        }
```

### Performance Characteristics
- **Cross-Validation**: Parallelized across CPU cores using `n_jobs=-1`
- **Random Forest Models**: Parallel training for both classification and regression (`n_jobs=-1`)
- **Linear Models**: Parallel computation for LogisticRegression and LinearRegression (`n_jobs=-1`)
- **Gradient Boosting Models**: Sequential by nature (no `n_jobs` parameter available)
- **Data Processing**: Standard pandas/sklearn performance patterns

## 🧪 Testing Architecture

### Test Structure
```
tests/
├── test_pipeline.py        # Pipeline integration tests (29 tests)
├── test_preprocessing.py   # Preprocessing unit tests (29 tests)
├── test_utils.py          # Utility function tests (17 tests)
└── __init__.py            # Test package initialization
```

### Testing Strategies Used

**1. Fixture-Based Testing:**
```python
class TestNumericalTransformer:
    @pytest.fixture
    def sample_numerical_data(self):
        """Reusable test data."""
        return pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
        })

    def test_fit_method(self, sample_numerical_data):
        """Test transformer fitting with fixture data."""
        transformer = NumericalTransformer()
        result = transformer.fit(sample_numerical_data)
        assert result is transformer
        assert transformer.is_fitted_
```

**2. Edge Case Testing:**
```python
def test_transform_before_fit(self, sample_numerical_data):
    """Test error handling for unfitted transformer."""
    transformer = NumericalTransformer()
    with pytest.raises(ValueError, match="Transformer must be fitted"):
        transformer.transform(sample_numerical_data)

def test_fit_empty_dataframe(self):
    """Test handling of edge case inputs."""
    transformer = NumericalTransformer()
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="Cannot fit on empty DataFrame"):
        transformer.fit(empty_df)
```

**3. Integration Testing:**
```python
def test_end_to_end_workflow_classification(self, classification_data):
    """Test complete pipeline workflow."""
    X, y = classification_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    pipeline = AutoMLPipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    score = pipeline.score(X_test, y_test)

    assert len(predictions) == len(X_test)
    assert score > 0
```

### Test Coverage Analysis
- **Total Tests**: 75 tests across all modules (29 + 29 + 17)
- **Pipeline Tests**: End-to-end workflow and configuration testing
- **Preprocessing Tests**: Individual transformer and data processing testing
- **Utility Tests**: Data validation and helper function testing
- **Test Categories**:
  - Unit tests for individual components
  - Integration tests for complete workflows
  - Edge case and error handling tests

## 🔧 Error Handling and Validation

### Actual Error Handling Implementation
The library implements straightforward, effective error handling with two custom exceptions:

**Custom Exceptions:**
```python
class DataValidationError(Exception):
    """Exception raised for data validation errors."""
    pass

class PipelineError(Exception):
    """Exception raised for pipeline-related errors."""
    pass
```

### Data Validation Implementation
```python
def validate_data(X, y) -> Tuple[pd.DataFrame, pd.Series]:
    """Input validation with clear error messages."""
    # Check for None inputs
    if X is None or y is None:
        raise DataValidationError("Input X and y cannot be None")

    # Convert to pandas format
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    elif not isinstance(X, pd.DataFrame):
        raise DataValidationError("X must be a pandas DataFrame or numpy array")

    if isinstance(y, np.ndarray):
        y = pd.Series(y, name="target")
    elif not isinstance(y, pd.Series):
        raise DataValidationError("y must be a pandas Series or numpy array")

    # Basic validation checks
    if X.empty:
        raise DataValidationError("Input X cannot be empty")
    if len(y) == 0:
        raise DataValidationError("Input y cannot be empty")
    if len(X) != len(y):
        raise DataValidationError(f"X and y must have same samples. X: {len(X)}, y: {len(y)}")
    if len(X) < 2:
        raise DataValidationError("Need at least 2 samples for training")

    return X.reset_index(drop=True), y.reset_index(drop=True)
```

### Pipeline Error Handling
```python
class AutoMLPipeline:
    def fit(self, X, y):
        """Pipeline fitting with basic error wrapping."""
        try:
            X_df, y_series = validate_data(X, y)
            # Preprocessing and model selection logic...
            return self
        except Exception as e:
            raise PipelineError(f"Pipeline fitting failed: {str(e)}") from e
```

### Error Handling Strategy
- **Input Validation**: Comprehensive checks in `validate_data()` function
- **Type Conversion**: Automatic numpy to pandas conversion with validation
- **State Validation**: Check if components are fitted before use (e.g., `_check_is_fitted()`)
- **Error Wrapping**: Higher-level operations wrap lower-level errors with context
- **Clear Messages**: Descriptive error messages for common user mistakes

The error handling is designed to be simple, clear, and helpful for debugging without being overly complex.

## 📈 Actual Architecture Benefits

### Maintainability
- **Clear separation of concerns** through SOLID principles
- **Consistent interfaces** via abstract base classes
- **Comprehensive testing** with 75 test cases
- **Proper error handling** with custom exceptions

### Extensibility
- **Abstract base classes** enable new transformer types
- **Factory pattern** allows new model types
- **Strategy pattern** supports new problem types
- **Configuration-driven** behavior modification

### Code Quality
- **Type hints** for better IDE support and documentation
- **Dataclasses** for clean configuration management
- **Enums** for type-safe constants
- **Docstrings** for comprehensive documentation

This architecture provides a solid foundation for automated machine learning while maintaining simplicity and avoiding over-engineering. The implementation focuses on proven patterns and practical extensibility rather than theoretical complexity.
