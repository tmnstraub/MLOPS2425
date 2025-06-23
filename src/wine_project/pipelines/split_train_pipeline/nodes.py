"""
Node for splitting data into training and testing sets, instrumented with
Great Expectations for data validation.
"""

import logging
from typing import Any, Dict, Tuple, List

import pandas as pd
import great_expectations as ge
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def split_data_and_validate(
    data: pd.DataFrame, parameters: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Splits data into features and target training and test sets, then validates
    the integrity and properties of the resulting splits using Great Expectations.

    Args:
        data: DataFrame containing features and the target column.
        parameters: Parameters defined in parameters.yml. Expected keys include:
                    - target_column
                    - test_fraction
                    - random_state

    Returns:
        A tuple containing:
        - X_train: Training features.
        - X_test: Testing features.
        - y_train: Training target.
        - y_test: Testing target.
        - X_train.columns: The list of feature columns.
        
    Raises:
        RuntimeError: If the Great Expectations validation on the split data fails.
    """
    logger.info("Starting data splitting and validation node.")

    # --- 1. Core Splitting Logic (from original node) ---
    logger.info("Performing data split based on parameters.")
    
    # Basic assertion to check for nulls before splitting
    assert not data.isnull().values.any(), "Input data for splitting contains null values."

    target_column = parameters["target_column"]
    y = data[target_column]
    X = data.drop(columns=[target_column])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_fraction"], random_state=parameters["random_state"]
    )

    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}")
    
    # --- 2. Great Expectations Validation ---
    logger.info("Performing Great Expectations validation on the data splits.")
    
    # Convert pandas objects to GE DataFrames to use the validation API
    ge_x_train = ge.from_pandas(X_train)
    ge_x_test = ge.from_pandas(X_test)
    ge_y_train = ge.from_pandas(pd.DataFrame(y_train, columns=[target_column]))
    ge_y_test = ge.from_pandas(pd.DataFrame(y_test, columns=[target_column]))

    # Expectation 1: Check for label leakage in feature sets
    logger.info(f"Checking for target column ('{target_column}') in feature sets.")
    assert target_column not in X_train.columns, f"Target column {target_column} found in feature set"
    assert target_column not in X_test.columns, f"Target column {target_column} found in feature set"

    # Expectation 2: Ensure columns match between train and test feature sets
    logger.info("Verifying column schema consistency between X_train and X_test.")
    train_cols_sorted = sorted(X_train.columns)
    test_cols_sorted = sorted(X_test.columns)
    assert train_cols_sorted == test_cols_sorted, "Train and test feature columns do not match."
    
    # Expectation 3: Check for nulls in the final datasets
    logger.info("Checking for null values in all output datasets.")
    for col in train_cols_sorted:
        ge_x_train.expect_column_values_to_not_be_null(col)
        ge_x_test.expect_column_values_to_not_be_null(col)
    
    # Check target columns for nulls
    ge_y_train.expect_column_values_to_not_be_null(target_column)
    ge_y_test.expect_column_values_to_not_be_null(target_column)

    # Expectation 4: Verify the split ratio is approximately correct
    total_rows = len(data)
    expected_test_rows = int(total_rows * parameters["test_fraction"])
    tolerance = 0.05 * total_rows  # Allow 5% tolerance
    
    actual_test_rows = len(X_test)
    assert abs(actual_test_rows - expected_test_rows) <= tolerance, \
        f"Test set size {actual_test_rows} deviates more than {tolerance} rows from expected {expected_test_rows}"

    logger.info("All validations passed successfully.")
    
    # --- 3. Return Data for Downstream Nodes ---
    return X_train, X_test, y_train, y_test, list(X_train.columns)