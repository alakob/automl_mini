"""Setup script for AutoML Mini library."""

from pathlib import Path

from setuptools import find_packages, setup

# Read README file
readme_path = Path(__file__).parent / "README.md"
long_description = (
    readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
)

# Read requirements
requirements = [
    "pandas>=1.3.0",
    "numpy>=1.21.0",
    "scikit-learn>=1.0.0",
    "scipy>=1.7.0",
]

dev_requirements = [
    "pytest>=6.0.0",
    "pytest-cov>=2.10.0",
    "black>=21.0.0",
    "flake8>=3.8.0",
    "mypy>=0.910",
]

setup(
    name="automl-mini",
    version="1.0.0",
    author="AutoML Mini Team",
    author_email="team@automl-mini.com",
    description="A simplified automated machine learning library focusing on core functionality",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alakob/autoML_mini",
    # Package structure
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    # Dependencies
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "all": requirements + dev_requirements,
    },
    # Python version requirement
    python_requires=">=3.8",
    # Package classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    # Keywords for discoverability
    keywords=[
        "machine-learning",
        "automl",
        "automated-machine-learning",
        "data-science",
        "preprocessing",
        "model-selection",
        "scikit-learn",
    ],
    # Entry points (if needed for command-line tools)
    entry_points={
        "console_scripts": [
            # Example: "automl-mini=automl_mini.cli:main",
        ],
    },
    # Include additional files
    include_package_data=True,
    zip_safe=False,
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/alakob/autoML_mini/issues",
        "Source": "https://github.com/alakob/autoML_mini",
        "Documentation": "https://automl-mini.readthedocs.io/",
    },
)
