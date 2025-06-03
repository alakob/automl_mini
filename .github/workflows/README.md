# GitHub Actions CI/CD Pipeline

This directory contains the automated CI/CD workflows for the AutoML Mini project.

## 🔄 Workflows

### `ci.yml` - Main CI Pipeline
**Trigger**: Push/PR to `main` or `develop` branches
**Duration**: ~3-5 minutes

**What it does:**
- ✅ **Multi-Python Testing**: Tests on Python 3.8, 3.9, 3.10, and 3.11
- ✅ **Code Quality**: Format checking (black, isort), linting (ruff)
- ✅ **Type Checking**: MyPy type validation (informational)
- ✅ **Security Scanning**: Bandit security checks
- ✅ **Test Suite**: Runs all 75 tests with coverage reporting
- ✅ **Coverage Upload**: Sends coverage data to Codecov
- ✅ **Package Building**: Validates package can be built and distributed

### `docs.yml` - Documentation Deployment
**Trigger**: Push to `main` branch
**Duration**: ~2-3 minutes

**What it does:**
- ✅ **Generate Coverage**: Runs tests and creates HTML coverage reports
- ✅ **Convert Markdown**: Transforms all .md files to HTML
- ✅ **Create Navigation**: Builds a documentation site with navigation
- ✅ **Deploy to Pages**: Automatically publishes to GitHub Pages
- ✅ **Update Site**: Available at `https://alakob.github.io/automl_mini/`

### `status.yml` - Simple Status Check
**Trigger**: Push/PR to `main` or `develop` branches
**Duration**: ~30 seconds

**What it does:**
- ✅ **Status Indicator**: Provides a simple pass/fail status for branch protection

## 🛠️ Local Development

The CI pipeline leverages existing Makefile commands:

```bash
# Run all CI checks locally
make ci-test

# Individual checks
make check-format    # Code formatting
make check-types     # Type checking
make security        # Security scanning
make test-cov        # Tests with coverage
```

## 🔧 Configuration

### Required Secrets (Optional)
- `CODECOV_TOKEN`: For coverage reporting (optional, public repos work without)

### Branch Protection
Recommended branch protection rules for `main`:
- ✅ Require status checks: `CI / test`
- ✅ Require up-to-date branches
- ✅ Require linear history

## 📊 Status Badges

Current badge in README.md:
```markdown
[![CI](https://github.com/alakob/automl_mini/actions/workflows/ci.yml/badge.svg)](https://github.com/alakob/automl_mini/actions/workflows/ci.yml)
```

## 🚀 Future Enhancements

1. **Automated Releases**: Tag-triggered PyPI publishing
2. **Performance Testing**: Benchmark regression detection
3. **Documentation**: Auto-deploy docs on changes
4. **Dependency Updates**: Automated dependency management
5. **Multi-OS Testing**: Windows and macOS support

## 📊 Test Coverage Access

### **Live Coverage Reports**
Interactive test coverage reports are automatically generated and deployed to GitHub Pages:

**Coverage URLs:**
- **Main Coverage**: `https://alakob.github.io/automl_mini/coverage/`
- **Navigation Hub**: `https://alakob.github.io/automl_mini/nav.html` (includes coverage link)

### **What You Get:**
- ✅ **Overall Coverage Percentage**: Summary of total code coverage
- ✅ **File-by-File Breakdown**: Coverage statistics for each Python file
- ✅ **Line-by-Line Analysis**: Interactive view showing which lines are covered
- ✅ **Missing Coverage**: Highlighted lines that need test coverage
- ✅ **Function Coverage**: Individual function coverage statistics

### **Updates Automatically**
- Coverage reports update every time you push to `main`
- Always reflects the latest test coverage from CI
- No manual deployment needed
