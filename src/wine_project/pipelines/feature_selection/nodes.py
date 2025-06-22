# src/wine_project/pipelines/feature_selection/nodes.py

import logging
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

# Note: Removed unused imports like os, pickle, great_expectations

# --- MODIFIED RFE FUNCTION ---
def feature_selection_rfe(X_train: pd.DataFrame, y_train: pd.DataFrame,
                          champion_model: Any, parameters: Dict[str, Any]) -> List[str]:
    """Performs feature selection using RFE with a given model."""
    log = logging.getLogger(__name__)
    log.info(f"Starting RFE feature selection with {len(X_train.columns)} columns.")

    # The model is now passed in directly! No file loading.
    classifier = champion_model
    log.info("Using pre-loaded champion model for RFE.")

    # It's good practice to handle the case where the model might not be a classifier
    if not hasattr(classifier, "fit") or not hasattr(classifier, "predict"):
        log.error("The provided 'champion_model' is not a valid classifier. Cannot perform RFE.")
        # Return all columns as a safe fallback
        return X_train.columns.tolist()

    y_train_flat = np.ravel(y_train)

    rfe = RFE(estimator=classifier) # Use the passed-in model
    rfe = rfe.fit(X_train, y_train_flat)
    
    selected_features_mask = rfe.support_
    selected_features = X_train.columns[selected_features_mask].tolist()

    log.info(f"Number of best columns after RFE: {len(selected_features)}")
    return selected_features

# --- CORRELATION FUNCTION (No changes needed, it's already good!) ---
def correlation_feature_selection(X_train: pd.DataFrame, y_train: pd.DataFrame, parameters: Dict[str, Any]) -> List[str]:
    # ... your existing code for this function is fine ...
    # (I've omitted it for brevity)
    log = logging.getLogger(__name__)
    log.info(f"Starting correlation-based feature selection with {len(X_train.columns)} columns.")
    # ... rest of the function ...
    correlation_threshold = parameters.get("correlation_threshold", 0.9)
    # ...
    correlation_matrix = X_train.corr(method='pearson').abs()
    # ...
    upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    # ...
    to_drop = set()
    # ...
    X_cols_corr = [col for col in X_train.columns if col not in to_drop]
    log.info(f"Number of columns after correlation-based feature selection: {len(X_cols_corr)}")
    return X_cols_corr


# --- NEW INTERSECTION NODE ---
def intersect_features(rfe_features: List[str], corr_features: List[str]) -> List[str]:
    """Finds the intersection of features from two lists."""
    log = logging.getLogger(__name__)
    
    final_selected_features = list(set(rfe_features) & set(corr_features))
    
    log.info(f"Features selected by RFE: {len(rfe_features)}")
    log.info(f"Features selected by Correlation: {len(corr_features)}")
    log.info(f"Final selected features (intersection): {len(final_selected_features)}")

    if not final_selected_features:
        log.warning("No common features found. Falling back to RFE features.")
        return rfe_features

    return final_selected_features