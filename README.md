![AutoML Logo](images/logo.png "AutoML Logo")

# AutoML Simple

> A streamlined automated machine learning library focusing on core functionality and best practices

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/alakob/automl_simple/actions/workflows/ci.yml/badge.svg)](https://github.com/alakob/automl_simple/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-75%20passing-green.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-high-green.svg)](tests/)
[![PEP 8](https://img.shields.io/badge/code%20style-PEP%208-blue.svg)](https://pep8.org/)
[![Type Hints](https://img.shields.io/badge/typing-PEP%20484-blue.svg)](https://peps.python.org/pep-0484/)
[![SOLID](https://img.shields.io/badge/principles-SOLID-blue.svg)](#architecture)

## 🎯 Overview

AutoML Simple is a Python library designed for rapid prototyping and implementation of automated machine learning workflows. Built with **engineering best practices** in mind, it demonstrates clean architecture, comprehensive testing, and production-ready code quality following Python standards.

### Key Features

- **🔄 Automated Pipeline**: End-to-end ML workflow automation
- **🔧 Preprocessing**: Intelligent handling of mixed data types (numerical & categorical)
- **🤖 Model Selection**: Cross-validated comparison of multiple algorithms
- **📊 Evaluation**: Comprehensive performance metrics and reporting
- **🏗️ SOLID Principles**: Clean, extensible, and maintainable architecture with proven design patterns
- **✅ Comprehensive Testing**: 75 pytest-based unit and integration tests with high coverage
- **📝 Documentation**: Clear usage examples and library reference following PEP 257
- **⚡ Modern Tooling**: Fast development with uv, pre-commit hooks, and automated quality checks
- **🐍 Python Standards**: PEP 8 compliant code with type hints (PEP 484) and modern Python practices

## 🚀 Quick Start

```bash
# 1. Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install AutoML Simple from GitHub
uv pip install "automl_simple @ git+https://github.com/alakob/automl_simple.git"

# 3. Use it!
```

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from automl_simple import AutoMLPipeline

# Load data and train
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

# AutoML in 3 lines
pipeline = AutoMLPipeline()
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

print(f"Test Accuracy: {pipeline.score(X_test, y_test):.3f}")
```

**That's it!** AutoML Simple handles preprocessing, model selection, and evaluation automatically.

For more detailed examples and configuration options, see the [Usage Examples](#-usage-examples) section below.

<details>
<summary>📋 Table of Contents</summary>

- [Installation](#-installation)
- [Usage Examples](#-usage-examples)
- [Architecture](#️-architecture)
- [Testing](#-testing)
- [Python Best Practices](#-python-best-practices)
- [Performance & Capabilities](#-performance--capabilities)
- [Library Reference](#️-library-reference)
- [Development](#-development)
- [Contributing](#-contributing)

</details>

## 📦 Installation

### Standard Installation
```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install from GitHub
uv pip install "automl_simple @ git+https://github.com/alakob/automl_simple.git"
```

### Development Installation
```bash
git clone https://github.com/alakob/automl_simple.git
cd automl_simple
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

<details>
<summary>🔧 Alternative Methods & Troubleshooting</summary>

#### Using pip (traditional)
```bash
python -m venv .venv
source .venv/bin/activate
pip install "automl_simple @ git+https://github.com/alakob/automl_simple.git"
```

#### Prerequisites
- Python 3.8 - 3.11
- Git for cloning repository

#### Troubleshooting
- **uv not found**: Restart shell or `export PATH="$HOME/.cargo/bin:$PATH"`
- **Permission errors**: Use virtual environments (recommended)

</details>

## 📚 Usage Examples

### Basic Usage
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

from automl_simple import AutoMLPipeline

# Load the iris dataset (following scikit-learn naming conventions)
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train and predict
pipeline = AutoMLPipeline()
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

# View results
results = pipeline.get_results()
print(results.summary())

# Additional information
print(f"Best model: {results.best_model}")
print(f"Test accuracy: {pipeline.score(X_test, y_test):.3f}")

# Feature importance (formatted)
print(pipeline.format_feature_importance())

# Or get raw feature importance as dictionary
# print(f"Feature importance: {pipeline.get_feature_importance()}")
```

### Custom Configuration
```python
from automl_simple import AutoMLPipeline, PipelineConfig, ProblemType

# Advanced configuration following Python best practices
config = PipelineConfig(
    cv_folds=5,
    problem_type=ProblemType.CLASSIFICATION,
    verbose=True,
    random_state=42
)

pipeline = AutoMLPipeline(config=config)
```

### Run Complete Examples
A comprehensive example demonstrating all library features is available in the `examples/` directory:

```bash
cd examples
python basic_usage.py
```

The `basic_usage.py` script includes:
- **Classification Example**: Generated dataset with mixed numeric and categorical features
- **Regression Example**: Diabetes progression dataset with feature engineering
- **Real-world Example**: Complete Iris classification workflow
- **Error Handling Demo**: Input validation and error handling showcase

## 🏗️ Architecture

The library follows **SOLID principles** and implements proven **design patterns** for maintainable and extensible code, adhering to Python best practices:

### Core Components

```
src/automl_simple/
├── __init__.py           # Public Interface (PEP 257 docstrings)
├── pipeline.py           # Main orchestration (AutoMLPipeline)
├── preprocessing.py      # Data preprocessing transformers
├── models.py            # Model selection and evaluation
└── utils.py             # Utilities and validation
```

### Python Standards Compliance
- **PEP 8**: Code style and formatting with black and isort
- **PEP 257**: Comprehensive docstring conventions
- **PEP 484**: Type hints throughout the codebase
- **PEP 518**: Modern `pyproject.toml` configuration
- **Import Organization**: Grouped imports (stdlib, third-party, local)
- **Naming Conventions**: snake_case for variables, PascalCase for classes

### Design Patterns Implemented
- **Factory Pattern**: Model creation for different problem types (Classification/Regression)
- **Strategy Pattern**: Problem type selection and algorithm strategy
- **Template Method Pattern**: Transformer workflow with ABC inheritance

### Key Classes

- **`AutoMLPipeline`**: Main orchestrator coordinating the entire workflow
- **`DataPreprocessor`**: Automatic feature type detection and preprocessing
- **`ModelSelector`**: Cross-validated model comparison and selection

> 📖 **Detailed Architecture**: See [ARCHITECTURE.md](ARCHITECTURE.md) for comprehensive design patterns and SOLID principles implementation.

## 🧪 Testing

The library includes comprehensive testing with **75 test cases** following Python testing best practices:

**Test Distribution:**
- **Pipeline Tests (29 tests)**: End-to-end workflow and integration testing
- **Preprocessing Tests (29 tests)**: Individual component testing (transformers, data processing)
- **Utility Tests (17 tests)**: Input validation, helper functions, and error handling scenarios

### Python Testing Standards
- **pytest**: Modern Python testing framework
- **Test Coverage**: High coverage with detailed reporting
- **Test Organization**: Clear separation of unit vs integration tests
- **Fixtures**: Reusable test data and configurations
- **Parameterized Tests**: Data-driven testing for multiple scenarios

### Running Tests

```bash
# Run all tests with coverage
uv run python -m pytest tests/ --cov=automl_simple

# Run specific test modules
uv run python -m pytest tests/test_pipeline.py -v

# Generate HTML coverage report
uv run python -m pytest tests/ --cov=automl_simple --cov-report=html
```

### Test Results
```
============= 75 passed in 11.09s =============
All tests passing with comprehensive coverage
```

## 🐍 Python Best Practices

This library demonstrates and follows established Python standards:

### Code Quality
- **Linting**: ruff for fast Python linting
- **Formatting**: black for consistent code formatting
- **Import Sorting**: isort for organized imports
- **Type Checking**: Type hints with mypy compatibility
- **Pre-commit Hooks**: Automated quality checks

### Development Standards
- **Virtual Environments**: Proper isolation with venv/uv
- **Dependency Management**: Modern pyproject.toml configuration
- **Version Control**: Meaningful commit messages and branch strategy
- **Documentation**: Comprehensive docstrings following PEP 257

### Performance Considerations
- **Lazy Loading**: Efficient memory usage patterns
- **Type Hints**: Better IDE support and runtime optimization
- **Error Handling**: Proper exception hierarchy and handling
- **Resource Management**: Context managers where appropriate

## 📈 Performance & Capabilities

### Performance Features
- **Parallel Cross-Validation**: All model evaluation uses `n_jobs=-1` for parallel processing
- **Parallel Model Training**: Random Forest and Linear models (LogisticRegression, LinearRegression) support parallel training
- **Sequential Models**: Gradient Boosting models run sequentially (inherent algorithm limitation)
- **Efficient Libraries**: Built on NumPy/Pandas optimized operations
- **Memory Management**: Proper DataFrame operations with index management

### Supported Problem Types
- **Classification**: Multi-class classification with probability estimates
- **Regression**: Continuous target prediction

### Algorithms Included
- **Random Forest**: Robust ensemble method
- **Logistic/Linear Regression**: Linear baseline models
- **Gradient Boosting**: Advanced boosting algorithms

### Preprocessing Features
- **Numerical**: Mean imputation + StandardScaler
- **Categorical**: Mode imputation + OneHot/Label encoding
- **Automatic type detection**: No manual specification needed
- **Missing value handling**: Robust imputation strategies

### Evaluation Metrics
- **Classification**: F1-score (weighted), accuracy, precision, recall
- **Regression**: R², RMSE, MAE
- **Cross-validation**: K-fold validation for reliable estimates

## 🎛️ Library Reference

### AutoMLPipeline

**Type-hinted API following PEP 484:**

```python
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

class AutoMLPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None) -> None: ...
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'AutoMLPipeline': ...
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray: ...
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray: ...  # Classification only
    def score(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> float: ...
    def get_results(self) -> PipelineResult: ...
    def get_feature_importance(self) -> Optional[Dict[str, float]]: ...
    def format_feature_importance(self, top_n: Optional[int] = None) -> Optional[str]: ...
```

### Configuration Classes

```python
@dataclass
class PipelineConfig:
    cv_folds: int = 3
    test_size: float = 0.2
    random_state: int = 42
    problem_type: Optional[ProblemType] = None
    verbose: bool = False

class ProblemType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
```

### Result Objects

```python
@dataclass
class PipelineResult:
    best_model: BaseEstimator
    best_score: float
    problem_type: ProblemType
    model_results: List[ModelResult]
    total_time: float
    # ... additional fields with proper type annotations
```

**Python Conventions:**
- **snake_case**: All variable and function names
- **PascalCase**: Class names (AutoMLPipeline, PipelineConfig)
- **Type Hints**: Complete type annotations for better IDE support
- **Dataclasses**: Modern Python data structures with automatic methods
- **Enums**: Type-safe constants for problem types
