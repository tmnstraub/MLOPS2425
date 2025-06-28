import numpy as np
import pandas as pd
import pytest
import warnings
import mlflow,yaml
import logging

warnings.filterwarnings("ignore", category=Warning)

from catboost import CatBoostRegressor
from src.wine_project.pipelines.model_train.nodes import model_train

logger = logging.getLogger(__name__)


def test_model_train():
    """
    Test that the model train node returns a model with a score
    """
    # Create dummy data
    X_train, X_val, X_train_encoded, X_val_encoded, y_train, y_val = create_dummy_data()

    # Define parameters
    parameters_false = {
        'use_feature_selection': False
    }

    parameters_true = {
        'use_feature_selection': True
    }

    best_columns = X_train.columns.to_list()  # Assuming no feature selection for this test

    # # Create a champion model placeholder
    # champion_dict = {'classifier': CatBoostRegressor, 'test_score': 0}
    # champion_model = None

    # Run the model train node without feature selection
    model, columns, results_dict, plot = model_train(X_train, X_val, y_train, y_val, parameters_false, best_columns)

    # Check that the returned value is a CatBoostRegressor
    assert isinstance(model, CatBoostRegressor)
    assert isinstance(results_dict, dict)
    assert len(columns) > 0

    #Run the model train node with feature selection
    model_fs, columns_fs, results_dict_fs, plot_fs = model_train(X_train, X_val, y_train, y_val, parameters_true, best_columns)

    # Check that the returned value is a CatBoostRegressor
    assert isinstance(model_fs, CatBoostRegressor)
    assert isinstance(results_dict_fs, dict)
    assert len(columns_fs) == len(best_columns)
    
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