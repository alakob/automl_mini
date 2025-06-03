# Contributing to AutoML Mini

Thank you for your interest in contributing to AutoML Mini! This document provides guidelines for development and contribution.

## 🚀 Quick Setup

```bash
# Clone and setup development environment
git clone https://github.com/alakob/autoML.git
cd autoML

# Create virtual environment with uv
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install in development mode with all dependencies
uv pip install -e ".[dev]"
```

## 🧪 Running Tests

```bash
# Run all tests
uv run python -m pytest tests/ -v

# Run with coverage
uv run python -m pytest tests/ --cov=automl_mini

# Generate HTML coverage report
uv run python -m pytest tests/ --cov=automl_mini --cov-report=html
open htmlcov/index.html  # View coverage report
```

## 🏗️ Development Standards

### Architecture Principles
- **SOLID Principles**: Follow Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, and Dependency Inversion
- **Design Patterns**: Utilize Factory, Strategy, and Template Method patterns as implemented in the codebase
- **Abstract Base Classes**: Use ABC for extensible interfaces (not Protocols)
- **Clear Separation of Concerns**: Each component has a single, well-defined responsibility

### Code Quality
- **Type Hints**: Use type annotations for all function signatures
- **Docstrings**: Comprehensive documentation for all public interfaces
- **Error Handling**: Implement graceful error handling with custom exceptions
- **Testing**: Maintain comprehensive test coverage (75+ tests currently)

### Testing Requirements
- **Unit Tests**: Test individual components in isolation (preprocessing, models, utils)
- **Integration Tests**: Test end-to-end pipeline workflows
- **Edge Cases**: Test input validation and error handling scenarios
- **Fixture-Based**: Use pytest fixtures for reusable test data

### Code Style
- **Consistent Formatting**: Use Black, isort, and Ruff as configured in pyproject.toml
- **Clear Naming**: Use descriptive variable and function names with type hints
- **Modular Design**: Keep functions and classes focused and small
- **Documentation**: Comment complex logic and design decisions

## 🔧 Adding New Features

### Extending Transformers (Template Method Pattern)
Follow the existing `BaseTransformer` abstract base class:

```python
from abc import ABC, abstractmethod
import pandas as pd

class CustomTransformer(BaseTransformer):
    """Example custom transformer following Template Method pattern."""

    def fit(self, X: pd.DataFrame) -> 'CustomTransformer':
        """Implement specific fitting logic."""
        # Your custom fitting logic here
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Implement specific transformation logic."""
        self._check_is_fitted()  # Template method validation
        # Your custom transformation logic here
        return transformed_X

# Add to DataPreprocessor detection logic in preprocessing.py
```

### Adding New Model Types (Factory Pattern)
Extend the existing factory pattern in `models.py`:

```python
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator

class CustomModelFactory(BaseModelFactory):
    """Example factory for custom model types."""

    def create_models(self) -> Dict[str, BaseEstimator]:
        """Create models appropriate for your problem type."""
        return {
            'custom_model_1': YourCustomModel(),
            'custom_model_2': AnotherCustomModel(),
        }

    def get_scoring_metric(self) -> str:
        """Return appropriate scoring metric."""
        return 'your_custom_metric'

# Update ModelSelector._get_model_factory() strategy mapping
```

### Adding New Problem Types (Strategy Pattern)
Extend the existing strategy pattern:

```python
# 1. Extend ProblemType enum in models.py
class ProblemType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CUSTOM_TYPE = "custom_type"  # Your new type

# 2. Create corresponding factory
class CustomProblemFactory(BaseModelFactory):
    # Implementation as shown above

# 3. Update strategy mapping in ModelSelector
def _get_model_factory(self, problem_type: ProblemType) -> BaseModelFactory:
    if problem_type == ProblemType.CLASSIFICATION:
        return ClassificationModelFactory()
    elif problem_type == ProblemType.REGRESSION:
        return RegressionModelFactory()
    elif problem_type == ProblemType.CUSTOM_TYPE:
        return CustomProblemFactory()
    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")
```

### Testing New Features
- **Unit Tests**: Test your component in isolation
- **Integration Tests**: Test with AutoMLPipeline end-to-end
- **Edge Cases**: Test error conditions and input validation
- **Existing Tests**: Ensure all 75+ existing tests still pass

## 📊 Performance Considerations

### Current Performance Features
- **Parallel Processing**: Models use `n_jobs=-1` for sklearn operations
- **Efficient Libraries**: Built on NumPy/Pandas optimized operations
- **Memory Management**: Proper DataFrame operations with index management

### When Adding Features
- Maintain sklearn's parallel processing capabilities
- Use efficient pandas operations
- Consider memory usage for large datasets
- Test performance impact with benchmarks

## 🚀 Development Workflow with Modern Python Tooling

### Pre-commit Hooks (Configured)
```bash
# Install pre-commit hooks (already configured in .pre-commit-config.yaml)
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

### Development Tools (Already Configured in pyproject.toml)
```bash
# Code formatting
black src/ tests/
isort src/ tests/

# Linting
ruff check src/ tests/
mypy src/

# Security scanning
bandit -r src/

# All checks via Makefile
make lint
make test
make format
```

### Development Workflow Benefits
- **Faster Installation**: 10x+ speedup compared to pip with uv
- **Better Dependency Resolution**: Prevents conflicts
- **Consistent Environments**: Reproducible builds
- **Automated Quality**: Pre-commit hooks ensure code quality

## 📋 Pull Request Guidelines

1. **Create Feature Branch**: `git checkout -b feature/your-feature-name`
2. **Follow Architecture**: Use existing design patterns (Factory, Strategy, Template Method)
3. **Add Tests**: Include unit and integration tests
4. **Update Documentation**: Update docstrings and examples
5. **Quality Checks**: Ensure all pre-commit hooks pass
6. **Verify All Tests**: Run full test suite (75+ tests)
7. **Clear Description**: Provide detailed PR description with context

## 🐛 Bug Reports

When reporting bugs, please include:
- Python version and environment details
- Minimal reproducible example using AutoMLPipeline
- Expected vs actual behavior
- Full error traceback if applicable
- Steps to reproduce the issue

## 💡 Feature Requests

For new features, consider:
- **Use Case**: Describe the motivation and real-world scenario
- **Architecture Fit**: How it aligns with existing SOLID principles
- **Design Pattern**: Which existing pattern it would extend (Factory/Strategy/Template)
- **Backward Compatibility**: Impact on existing interfaces
- **Implementation**: Rough approach and extension points to use

## 📚 Code Examples and Common Patterns

### Configuration Extension
```python
@dataclass
class CustomPipelineConfig(PipelineConfig):
    """Extend existing configuration."""
    custom_parameter: int = 10
    custom_flag: bool = False
```

### Error Handling Pattern
```python
class CustomTransformerError(Exception):
    """Custom exception for your transformer."""
    pass

def your_method(self, X: pd.DataFrame):
    try:
        # Your implementation
        return result
    except Exception as e:
        raise CustomTransformerError(f"Custom operation failed: {str(e)}")
```

## 🏆 Recognition

Contributors who follow these guidelines help maintain AutoML Mini as a high-quality, maintainable library that demonstrates production-ready development practices with proper SOLID principles and proven design patterns.
