"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split



def split_data(
    data: pd.DataFrame, parameters: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits data into features and target training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters.yml.
    Returns:
        Split data.
    """

    assert [col for col in data.columns if data[col].isnull().any()] == []
    y = data[parameters["target_column"]]
    X = data.drop(columns=parameters["target_column"], axis=1)
    X = X.drop(columns="index", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=parameters["test_fraction"], random_state=parameters["random_state"])

    return X_train, X_test, y_train, y_test, X_train.columns


