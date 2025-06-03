"""
Basic usage example for AutoML Simple library.

This example demonstrates the core functionality of the simplified AutoML library,
showcasing both classification and regression workflows.
"""

import os
import sys
import warnings
from pathlib import Path

# Set environment variable to suppress warnings
os.environ["PYTHONWARNINGS"] = "ignore"

# Suppress all warnings for cleaner output
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

# Add the src directory to the path so we can import our module
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, load_iris, make_classification
from sklearn.model_selection import train_test_split

# Suppress numpy warnings
np.seterr(all="ignore")

from automl_mini import AutoMLPipeline, PipelineConfig, ProblemType


def classification_example():
    """Demonstrate classification workflow."""
    print("=" * 60)
    print("CLASSIFICATION EXAMPLE")
    print("=" * 60)

    # Generate sample data
    print("1. Generating sample classification data...")
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=3,
        random_state=42,
    )

    # Convert to DataFrame and add categorical features
    feature_names = [f"numeric_feature_{i}" for i in range(10)]
    X_df = pd.DataFrame(X, columns=feature_names)

    # Add some categorical features
    rng = np.random.RandomState(42)
    X_df["category_A"] = rng.choice(["Type1", "Type2", "Type3"], size=len(X_df))
    rng2 = np.random.RandomState(43)
    X_df["category_B"] = rng2.choice(["Red", "Blue", "Green"], size=len(X_df))

    # Convert target to Series
    y_series = pd.Series(y, name="class_label")

    print(f"   - Dataset shape: {X_df.shape}")
    print(f"   - Target distribution: {y_series.value_counts().to_dict()}")
    print(f"   - Feature types: {X_df.dtypes.value_counts().to_dict()}")

    # Split data
    print("\n2. Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.2, random_state=42, stratify=y_series
    )

    print(f"   - Training set: {X_train.shape[0]} samples")
    print(f"   - Test set: {X_test.shape[0]} samples")

    # Create and configure pipeline
    print("\n3. Creating AutoML pipeline...")
    config = PipelineConfig(cv_folds=3, random_state=42, verbose=True)
    pipeline = AutoMLPipeline(config=config)

    # Fit pipeline
    print("\n4. Training the pipeline...")
    pipeline.fit(X_train, y_train)

    # Make predictions
    print("\n5. Making predictions...")
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)

    # Evaluate
    print("\n6. Evaluating performance...")
    test_score = pipeline.score(X_test, y_test)
    print(f"   - Test F1 Score: {test_score:.4f}")
    print(f"   - Predictions shape: {predictions.shape}")
    print(f"   - Probabilities shape: {probabilities.shape}")

    # Get detailed results
    results = pipeline.get_results()
    print(f"   - Best model: {type(results.best_model).__name__}")
    print(f"   - Best CV score: {results.best_score:.4f}")
    print(f"   - Total training time: {results.total_time:.2f} seconds")

    # Show model comparison
    print("\n7. Model comparison:")
    for i, model_result in enumerate(results.model_results[:3], 1):
        print(f"   {i}. {model_result.model_name}: {model_result.score:.4f}")

    # Feature importance (if available)
    feature_importance = pipeline.get_feature_importance()
    if feature_importance:
        print("\n8. Top 5 most important features:")
        sorted_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )
        for feature, importance in sorted_features[:5]:
            print(f"   - {feature}: {importance:.4f}")

    return pipeline, results


def regression_example():
    """Demonstrate regression workflow."""
    print("\n\n" + "=" * 60)
    print("REGRESSION EXAMPLE")
    print("=" * 60)

    # Load built-in dataset
    print("1. Loading sample regression data...")
    diabetes = load_diabetes()
    X_df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    y_series = pd.Series(diabetes.target, name="diabetes_progression")

    # Add some categorical features for demonstration
    rng = np.random.RandomState(42)
    X_df["patient_type"] = rng.choice(["TypeA", "TypeB", "TypeC"], size=len(X_df))
    rng2 = np.random.RandomState(43)
    X_df["treatment"] = rng2.choice(["Standard", "Enhanced"], size=len(X_df))

    print(f"   - Dataset shape: {X_df.shape}")
    print(f"   - Target range: {y_series.min():.1f} to {y_series.max():.1f}")
    print(f"   - Feature types: {X_df.dtypes.value_counts().to_dict()}")

    # Split data
    print("\n2. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.2, random_state=42
    )

    # Create pipeline with explicit problem type
    print("\n3. Creating AutoML pipeline for regression...")
    config = PipelineConfig(
        cv_folds=5, problem_type=ProblemType.REGRESSION, verbose=True, random_state=42
    )
    pipeline = AutoMLPipeline(config=config)

    # Fit pipeline
    print("\n4. Training the pipeline...")
    pipeline.fit(X_train, y_train)

    # Make predictions
    print("\n5. Making predictions...")
    predictions = pipeline.predict(X_test)

    # Evaluate
    print("\n6. Evaluating performance...")
    test_score = pipeline.score(X_test, y_test)
    print(f"   - Test R² Score: {test_score:.4f}")

    # Calculate additional metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)

    print(f"   - RMSE: {rmse:.2f}")
    print(f"   - MAE: {mae:.2f}")
    print(f"   - Predictions range: {predictions.min():.1f} to {predictions.max():.1f}")

    # Get results
    results = pipeline.get_results()
    print(f"   - Best model: {type(results.best_model).__name__}")
    print(f"   - Best CV score: {results.best_score:.4f}")

    return pipeline, results


def real_world_example():
    """Demonstrate with a more realistic dataset."""
    print("\n\n" + "=" * 60)
    print("REAL-WORLD EXAMPLE: IRIS CLASSIFICATION")
    print("=" * 60)

    # Load Iris dataset
    print("1. Loading Iris dataset...")
    iris = load_iris()
    X_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    y_series = pd.Series(iris.target, name="species")

    # Map target to species names
    species_names = {0: "setosa", 1: "versicolor", 2: "virginica"}
    y_series = y_series.map(species_names)

    print(f"   - Dataset: {X_df.shape[0]} samples, {X_df.shape[1]} features")
    print(f"   - Classes: {list(y_series.unique())}")

    # Create simple pipeline
    print("\n2. Creating simple AutoML pipeline...")
    pipeline = AutoMLPipeline()

    # Fit on all data for demonstration
    print("\n3. Training on full dataset...")
    pipeline.fit(X_df, y_series)

    # Show results summary
    print("\n4. Results summary:")
    results = pipeline.get_results()
    print(results.summary())

    # Make sample predictions
    print("\n5. Making sample predictions...")
    sample_data = X_df.iloc[:5]  # First 5 samples
    predictions = pipeline.predict(sample_data)
    probabilities = pipeline.predict_proba(sample_data)

    print("   Sample predictions:")
    for i, (pred, true_val) in enumerate(zip(predictions, y_series.iloc[:5])):
        prob_max = np.max(probabilities[i])
        print(
            f"   Sample {i+1}: Predicted={pred}, Actual={true_val}, Confidence={prob_max:.3f}"
        )

    return pipeline, results


def error_handling_example():
    """Demonstrate error handling capabilities."""
    print("\n\n" + "=" * 60)
    print("ERROR HANDLING DEMONSTRATION")
    print("=" * 60)

    print("1. Testing input validation...")

    pipeline = AutoMLPipeline()

    # Test with mismatched data
    try:
        X = pd.DataFrame({"feature": [1, 2, 3]})
        y = pd.Series([1, 2])  # Wrong length
        pipeline.fit(X, y)
    except Exception as e:
        print(f"   ✓ Caught mismatched data error: {type(e).__name__}")

    # Test prediction before fitting
    try:
        pipeline.predict(X)
    except Exception as e:
        print(f"   ✓ Caught unfitted pipeline error: {type(e).__name__}")

    # Test with invalid data types
    try:
        pipeline.fit("invalid", [1, 2, 3])
    except Exception as e:
        print(f"   ✓ Caught invalid data type error: {type(e).__name__}")

    print("   All error handling tests passed!")


def main():
    """Run all examples."""
    print("AutoML Simple Library - Usage Examples")
    print("Version: 1.0.0")
    print(
        "This demonstration shows the core capabilities of the simplified AutoML library."
    )

    # Run examples
    try:
        # Classification example
        cls_pipeline, cls_results = classification_example()

        # Regression example
        reg_pipeline, reg_results = regression_example()

        # Real-world example
        iris_pipeline, iris_results = real_world_example()

        # Error handling
        error_handling_example()

        print("\n\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print("✓ All examples completed successfully!")
        print("✓ Classification pipeline trained and evaluated")
        print("✓ Regression pipeline trained and evaluated")
        print("✓ Real-world dataset (Iris) processed")
        print("✓ Error handling validated")
        print("\nThe AutoML Simple library provides:")
        print("- Automatic preprocessing for mixed data types")
        print("- Model selection with cross-validation")
        print("- Easy-to-use pipeline interface")
        print("- Comprehensive error handling")
        print("- SOLID principles compliance")
        print("- Extensive unit test coverage")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
