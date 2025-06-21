import logging
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE


# --- ORIGINAL FEATURE SELECTION FUNCTION (RFE) ---
def feature_selection(X_train: pd.DataFrame, y_train: pd.DataFrame, parameters: Dict[str, Any]) -> List[str]:
    """
    Performs feature selection using Recursive Feature Elimination (RFE)
    with a RandomForestClassifier. This function is designed to be used
    as part of a Kedro pipeline.

    Args:
        X_train (pd.DataFrame): The training feature set.
        y_train (pd.DataFrame): The training target variable.
        parameters (Dict[str, Any]): A dictionary of parameters, expected to contain:
            - "baseline_model_params" (dict): Parameters for the RandomForestClassifier.

    Returns:
        List[str]: A list of selected column names based on RFE.
    """
    log = logging.getLogger(__name__)
    log.info(f"Starting RFE feature selection with: {len(X_train.columns)} columns.")

    y_train = np.ravel(y_train)
    classifier = None
    # open pickle file with champion model (regressor/classifier)
    try:
        with open(os.path.join(os.getcwd(), 'data', '06_models', 'champion_model.pkl'), 'rb') as f:
            classifier = pickle.load(f)
        log.info("Loaded champion model for RFE.")
    except FileNotFoundError:
        log.warning("Champion model not found at 'data/06_models/champion_model.pkl'. "
                    "Initializing RandomForestClassifier for RFE.")
        classifier = RandomForestClassifier(**parameters.get('baseline_model_params', {}))
    except Exception as e:
        log.error(f"Error loading champion model for RFE: {e}. "
                  "Initializing RandomForestClassifier for RFE.")
        classifier = RandomForestClassifier(**parameters.get('baseline_model_params', {}))

    if classifier is None: # Fallback if classifier still isn't initialized
        log.error("Classifier could not be initialized for RFE. Using a default RandomForestClassifier.")
        classifier = RandomForestClassifier(random_state=42)


    rfe = RFE(classifier)
    rfe = rfe.fit(X_train, y_train)
    f = rfe.get_support(1)  # the most important features
    X_cols_rfe = X_train.columns[f].tolist()

    log.info(f"Number of best columns after RFE: {len(X_cols_rfe)}")

    return X_cols_rfe

# --- CORRELATION FEATURE SELECTION FUNCTION ---
def correlation_feature_selection(X_train: pd.DataFrame, y_train: pd.DataFrame, parameters: Dict[str, Any]) -> List[str]:
    """
    Performs feature selection based on a correlation matrix.

    This function identifies highly correlated features and removes one from each
    highly correlated pair, prioritizing the feature with a stronger correlation
    to the target variable (y_train).

    Args:
        X_train (pd.DataFrame): The training feature set.
        y_train (pd.DataFrame): The training target variable.
        parameters (Dict[str, Any]): A dictionary of parameters, expected to contain:
            - "correlation_threshold" (float): The threshold above which features
              are considered highly correlated (e.g., 0.9).

    Returns:
        List[str]: A list of selected column names after correlation-based feature selection.
    """
    log = logging.getLogger(__name__)
    log.info(f"Starting correlation-based feature selection with {len(X_train.columns)} columns.")

    # Ensure y_train is a Series for easier correlation calculation
    if isinstance(y_train, pd.DataFrame) and y_train.shape[1] == 1:
        y_train = y_train.squeeze()
    elif isinstance(y_train, np.ndarray):
        y_train = pd.Series(y_train.ravel())  # Ensure it's 1D for correlation

    # Get the correlation threshold from parameters
    correlation_threshold = parameters.get("correlation_threshold", 0.9)
    if not isinstance(correlation_threshold, (int, float)) or not (0 < correlation_threshold <= 1):
        log.warning(
            f"Invalid 'correlation_threshold' parameter: {correlation_threshold}. "
            "Using default value of 0.9."
        )
        correlation_threshold = 0.9

    # Calculate the Pearson correlation matrix for features
    correlation_matrix = X_train.corr(method='pearson').abs()

    # Calculate correlation of each feature with the target variable
    data_with_target = pd.concat([X_train, y_train.rename('target')], axis=1)
    feature_target_correlations = data_with_target.corr(method='pearson')['target'].abs().drop('target')

    # Select upper triangle of correlation matrix
    upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than the threshold
    to_drop = set()
    for i in range(len(upper_tri.columns)):
        for j in range(i + 1, len(upper_tri.columns)):
            col_i = upper_tri.columns[i]
            col_j = upper_tri.columns[j]

            if upper_tri.iloc[i, j] >= correlation_threshold:
                if col_i in to_drop and col_j in to_drop:
                    continue

                corr_i_target = feature_target_correlations.get(col_i, 0)
                corr_j_target = feature_target_correlations.get(col_j, 0)

                if corr_i_target >= corr_j_target:
                    if col_j not in to_drop:
                        to_drop.add(col_j)
                        log.info(f"Dropping '{col_j}' due to high correlation ({upper_tri.iloc[i, j]:.2f}) with '{col_i}'. "
                                 f"Keeping '{col_i}' (corr with target: {corr_i_target:.2f} vs {corr_j_target:.2f}).")
                else:
                    if col_i not in to_drop:
                        to_drop.add(col_i)
                        log.info(f"Dropping '{col_i}' due to high correlation ({upper_tri.iloc[i, j]:.2f}) with '{col_j}'. "
                                 f"Keeping '{col_j}' (corr with target: {corr_j_target:.2f} vs {corr_i_target:.2f}).")

    X_cols_corr = [col for col in X_train.columns if col not in to_drop]

    log.info(f"Number of columns after correlation-based feature selection: {len(X_cols_corr)}")

    return X_cols_corr

# --- NEW DISPATCHER FUNCTION TO RUN BOTH ---
def dispatch_feature_selection(X_train: pd.DataFrame, y_train: pd.DataFrame, parameters: Dict[str, Any]) -> List[str]:
    """
    Executes both RFE and correlation-based feature selection methods
    and returns the intersection of the selected features.

    Args:
        X_train (pd.DataFrame): The training feature set.
        y_train (pd.DataFrame): The training target variable.
        parameters (Dict[str, Any]): A dictionary of parameters, expected to contain:
            - "baseline_model_params" (dict, optional): Parameters for the base model in RFE.
            - "correlation_threshold" (float, optional): Threshold for correlation method.

    Returns:
        List[str]: A list of selected column names that are common to both methods.
    """
    log = logging.getLogger(__name__)
    log.info("Running both RFE and correlation-based feature selection.")

    # 1. Run RFE
    rfe_selected_features = feature_selection(X_train, y_train, parameters)

    # 2. Run Correlation-based selection
    corr_selected_features = correlation_feature_selection(X_train, y_train, parameters)

    # 3. Find the intersection of features from both methods
    final_selected_features = list(set(rfe_selected_features) & set(corr_selected_features))

    log.info(f"Features selected by RFE: {len(rfe_selected_features)}")
    log.info(f"Features selected by Correlation: {len(corr_selected_features)}")
    log.info(f"Final selected features (intersection): {len(final_selected_features)}")

    if not final_selected_features:
        log.warning("No common features found between RFE and correlation-based selection. "
                    "This might indicate an issue with thresholds or data. Returning RFE selected features as a fallback.")
        return rfe_selected_features # Fallback to RFE if no intersection

    return final_selected_features
