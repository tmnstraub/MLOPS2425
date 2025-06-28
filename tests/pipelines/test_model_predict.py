import numpy as np
import pandas as pd
import pytest
import warnings
import mlflow,yaml
import logging
import pickle

warnings.filterwarnings("ignore", category=Warning)

from catboost import CatBoostRegressor
from src.wine_project.pipelines.model_predict.nodes import model_predict

logger = logging.getLogger(__name__)


def test_model_predict():
    """
    Test that the model predict node returns predictions
    """
    # Create dummy data
    X_train, X_val, X_train_encoded, X_val_encoded, y_train, y_val = create_dummy_data()

    # Define columns to use for prediction
    production_columns = X_train.columns.tolist()  # Use all columns

    # Create and fit a simple model on the training data
    # Identify categorical columns by their dtype
    cat_features = [col for col in X_train.columns if X_train[col].dtype == 'object']
    cat_feature_indices = [X_train.columns.get_loc(col) for col in cat_features]
    
    # Create the model with categorical feature information
    production_model = CatBoostRegressor(iterations=10, verbose=False, cat_features=cat_feature_indices)
    production_model.fit(X_train, y_train)

    # Pickle the model and columns as would be done in production
    model_pickle = production_model  # No need to pickle for the test
    columns_pickle = X_train.columns.tolist()  # No need to pickle for the test
    
    # # Unpickle for testing (simulating loading from disk)
    # model_unpickled = pickle.loads(model_pickle)
    # columns_unpickled = pickle.loads(columns_pickle)

    # Run the model predict node
    predictions, predictions_description = model_predict(X_val, model_pickle, columns_pickle)

    # Verify predictions
    assert isinstance(predictions, pd.DataFrame)
    assert 'y_pred' in predictions.columns
    assert len(predictions) == len(X_val)
    assert isinstance(predictions_description, pd.DataFrame)
    assert not predictions_description.empty


    
def create_dummy_data():
        # One numerical feature
        X_train_numeric = pd.DataFrame(np.random.rand(100, 1), columns=['feat1'])
        X_val_numeric = pd.DataFrame(np.random.rand(50, 1), columns=['feat1'])
        
        # Four categorical features with random categories
        categories = {
            'feat2': ['A', 'B', 'C', 'D'],
            'feat3': ['red', 'white', 'rose'],
            'feat4': ['France', 'Italy', 'Spain', 'USA', 'Australia'],
            'feat5': ['dry', 'sweet', 'medium']
        }
        
        # Generate categorical data
        X_train_cat = pd.DataFrame({
            'feat2': np.random.choice(categories['feat2'], size=100),
            'feat3': np.random.choice(categories['feat3'], size=100),
            'feat4': np.random.choice(categories['feat4'], size=100),
            'feat5': np.random.choice(categories['feat5'], size=100)
        })
        
        X_val_cat = pd.DataFrame({
            'feat2': np.random.choice(categories['feat2'], size=50),
            'feat3': np.random.choice(categories['feat3'], size=50),
            'feat4': np.random.choice(categories['feat4'], size=50),
            'feat5': np.random.choice(categories['feat5'], size=50)
        })
        
        # Combine numerical and categorical features
        X_train = pd.concat([X_train_numeric, X_train_cat], axis=1)
        X_val = pd.concat([X_val_numeric, X_val_cat], axis=1)
        
        # Create a one-hot encoded version for models that need it
        X_train_encoded = pd.get_dummies(X_train, drop_first=False)
        X_val_encoded = pd.get_dummies(X_val, drop_first=False)
        
        # Ensure X_val_encoded has the same columns as X_train_encoded
        for col in X_train_encoded.columns:
            if col not in X_val_encoded.columns:
                X_val_encoded[col] = 0
        X_val_encoded = X_val_encoded[X_train_encoded.columns]

        # Create dummy target variables with random float values between 0 and 100
        y_train = pd.Series(np.random.rand(100) * 100, name='price')
        y_val = pd.Series(np.random.rand(50) * 100, name='price')
        
        return X_train, X_val, X_train_encoded, X_val_encoded, y_train, y_val