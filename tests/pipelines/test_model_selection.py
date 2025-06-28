import numpy as np
import pandas as pd
import pytest
import warnings
import mlflow,yaml
import logging

warnings.filterwarnings("ignore", category=Warning)

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from src.wine_project.pipelines.model_selection.nodes import model_selection

logger = logging.getLogger(__name__)

@pytest.mark.slow
def test_model_selection():
    """
    Test that the model selection node returns a model with a score
    """
    # Create dummy data with mainly categorical features
    X_train, X_val, X_train_encoded, X_val_encoded, y_train, y_val = create_dummy_data()
    
    champion_dict_loser = {'regressor': None, 'val_rmse': 10000}
    
    champion_model_loser = None

    champion_dict_winner = {'regressor': RandomForestRegressor(), 'val_rmse': 0.1} # Example of a good champion model

    champion_model_winner = RandomForestRegressor()
    
    parameters = {
        'hyperparameters': {
            'XGBRegressor': {'n_estimators': [10, 50, 100], 'max_depth': [3, 5, 10]},
            'CatBoostRegressor': {'learning_rate': [0.1, 0.01], 'n_estimators': [50, 100]},
        }
    }
    
    # Run the model selection node against losing champion model
    model = model_selection(X_train, X_val, X_train_encoded, X_val_encoded, y_train, y_val, champion_dict_loser, champion_model_loser, parameters)
    
    # Check that the returned value is a model instance
    assert isinstance(model, XGBRegressor) or isinstance(model, CatBoostRegressor)
    assert isinstance(model.score(X_val, y_val), float)

    # Run the model selection node against winning champion model
    model = model_selection(X_train, X_val, X_train_encoded, X_val_encoded, y_train, y_val, champion_dict_winner, champion_model_winner, parameters)

    # Check that the returned value is a model instance
    assert isinstance(model, RandomForestRegressor)
    
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
