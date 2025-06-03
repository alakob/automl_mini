.PHONY: help install install-dev test test-cov lint format check-format check-types security clean pre-commit setup-dev

# Default target
help:
	@echo "Available commands:"
	@echo "  install      Install package in development mode"
	@echo "  install-dev  Install package with development dependencies"
	@echo "  test         Run tests"
	@echo "  test-cov     Run tests with coverage"
	@echo "  lint         Run all linting checks"
	@echo "  format       Format code with black and isort"
	@echo "  check-format Check if code is formatted correctly"
	@echo "  check-types  Run type checking with mypy"
	@echo "  security     Run security checks with bandit"
	@echo "  pre-commit   Install and run pre-commit hooks"
	@echo "  setup-dev    Complete development setup"
	@echo "  clean        Clean up cache files and build artifacts"

# Installation
install:
	uv pip install -e .

install-dev:
	uv pip install -e ".[dev]"

# Testing
test:
	uv run python -m pytest tests/

test-cov:
	uv run python -m pytest tests/ --cov=automl_mini --cov-report=html --cov-report=term --cov-report=xml

# Code formatting
format:
	uv run black src/ tests/
	uv run isort src/ tests/

check-format:
	uv run black --check src/ tests/
	uv run isort --check-only src/ tests/

# Linting
lint: check-format check-types security
	uv run ruff check src/ tests/

check-types:
	uv run mypy src/

security:
	uv run bandit -r src/ -c pyproject.toml

# Pre-commit
pre-commit:
	uv run pre-commit install
	uv run pre-commit run --all-files

# Development setup
setup-dev: install-dev pre-commit
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to verify everything works."

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/
	rm -rf .coverage

# CI/CD commands
ci-test: lint test-cov
	@echo "All CI checks passed!"

# Quick development workflow
dev-check: format lint test
	@echo "Development checks complete!"
