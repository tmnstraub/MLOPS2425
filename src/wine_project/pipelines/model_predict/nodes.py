
import pandas as pd
import logging
from typing import Dict, Tuple, Any
import numpy as np  
import pickle

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

logger = logging.getLogger(__name__)

def model_predict(X: pd.DataFrame,
                  model: pickle.Pickler, columns: pickle.Pickler) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Predict using the trained model.

    Args:
    --
        X (pd.DataFrame): Serving observations.
        model (pickle): Trained model.

    Returns:
    --
        scores (pd.DataFrame): Dataframe with new predictions.
    """

    logger.info('Starting model prediction...')
    
    # Predict
    y_pred = model.predict(X[columns])

    logger.info('Model prediction completed.')

    # Create dataframe with predictions
    X['y_pred'] = y_pred
    
    # Create dictionary with predictions
    describe_servings = X.describe()

    

    logger.info('Service predictions created.')
    logger.info('#servings: %s', len(y_pred))
    return X, describe_servings