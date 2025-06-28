"""
Tests for the feature selection node.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

# Import the function to be tested
# Adjust the import path based on your project structure
# This should be the correct path for your project
from src.wine_project.pipelines.feature_selection.nodes import feature_selection


@pytest.fixture
def sample_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates a sample DataFrame for testing feature selection.

    This fixture generates a dataset where some features are intentionally made
    more predictive of the target than others. This allows us to have a
    reasonable expectation of which features should be selected.

    - `important_feature_1` and `important_feature_2` are highly correlated with the target.
    - `noise_feature_1` and `noise_feature_2` are random noise.
    - `categorical_feature` is included for CatBoost compatibility.

    Returns:
        A tuple containing sample X_train and y_train DataFrames.
    """
    n_samples = 100
    np.random.seed(42)  # for reproducibility

    X_data = {
        "noise_feature_1": np.random.rand(n_samples),
        "important_feature_1": np.linspace(0, 10, n_samples),
        "noise_feature_2": np.random.normal(0, 1, n_samples),
        "important_feature_2": np.linspace(-5, 5, n_samples) ** 2,
        "categorical_feature": pd.Series(
            np.random.choice(['A', 'B', 'C'], n_samples)
        ).astype("category"),
    }
    X_train = pd.DataFrame(X_data)

    # The target is a function of the important features plus some small noise
    y_target = (
        3 * X_train["important_feature_1"]
        + 2 * X_train["important_feature_2"]
        + np.random.normal(0, 0.1, n_samples)
    )
    y_train = pd.DataFrame(y_target, columns=["target"])

    return X_train, y_train


@pytest.mark.parametrize(
    "selection_method, expected_selected_features",
    [
        ("rfe", ["important_feature_1", "important_feature_2"]),
        ("tree_based", ["important_feature_1", "important_feature_2"]),
        ("catboost", ["important_feature_1", "important_feature_2", "categorical_feature"]),
    ],
)
# We use patch to prevent the function from trying to access the filesystem.
# By raising FileNotFoundError, we force the function to use its fallback logic,
# which is to instantiate a new model. This makes our test self-contained.
@patch('builtins.open', side_effect=FileNotFoundError)
def test_feature_selection(
    mock_open, sample_data, selection_method, expected_selected_features
):
    """
    Tests the feature_selection function for all available methods.

    Args:
        mock_open: A mock object for the built-in `open` function from unittest.mock.
        sample_data: The pytest fixture providing test data.
        selection_method: The feature selection method to test ('rfe', 'tree_based', 'catboost').
        expected_selected_features: A list of features we expect to be selected.
    """
    # 1. ARRANGE
    X_train, y_train = sample_data
    
    # Define parameters dictionary for the node
    # The baseline_model_params can be minimal as we use small data
    parameters = {
        "feature_selection": selection_method,
        "baseline_model_params": {
            "random_state": 42,
            "n_estimators": 10, # Use few estimators for speed
            "verbose": 0
        },
    }

    # 2. ACT
    selected_cols = feature_selection(X_train, y_train, parameters)

    # 3. ASSERT
    
    # Basic structural checks
    assert isinstance(selected_cols, list), "Output should be a list"
    assert len(selected_cols) > 0, "Should select at least one feature"
    assert all(isinstance(col, str) for col in selected_cols), "All items in the list should be strings"
    
    # Check that selected columns are a subset of the original columns
    assert set(selected_cols).issubset(set(X_train.columns)), \
        "Selected features must be a subset of original features"

    # Core logic check: ensure the most important features are selected
    # We use a set for order-independent comparison.
    assert set(expected_selected_features).issubset(set(selected_cols)), \
        f"For '{selection_method}', expected features were not selected."

    # Check that noisy features are NOT selected
    assert "noise_feature_1" not in selected_cols
    assert "noise_feature_2" not in selected_cols
    
    # Ensure the mock was called, confirming we avoided filesystem access
    # This check is more relevant for 'rfe' and 'catboost' which have the try/except block
    if selection_method in ["rfe", "catboost"]:
        mock_open.assert_called()


def test_feature_selection_returns_list_of_strings(sample_data):
    """
    A specific test to ensure the output is always a list of strings,
    even if something unexpected happens.
    """
    # ARRANGE
    X_train, y_train = sample_data
    parameters = {
        "feature_selection": "tree_based",
        "baseline_model_params": {},
    }
    
    # ACT
    result = feature_selection(X_train, y_train, parameters)

    # ASSERT
    assert isinstance(result, list)
    assert all(isinstance(item, str) for item in result)