
import pandas as pd
import logging
from typing import Dict, Tuple, Any
import numpy as np  
import yaml
import pickle
import warnings
warnings.filterwarnings("ignore", category=Warning)

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from math import sqrt

import mlflow

logger = logging.getLogger(__name__)

def _get_or_create_experiment_id(experiment_name: str) -> str:
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        logger.info(f"Experiment '{experiment_name}' not found. Creating new one.")
        return mlflow.create_experiment(experiment_name)
    return exp.experiment_id
     
def model_selection(X_train: pd.DataFrame, 
                    X_test: pd.DataFrame, 
                    y_train: pd.DataFrame, 
                    y_test: pd.DataFrame,
                    champion_dict: Dict[str, Any],
                    champion_model : pickle.Pickler,
                    parameters: Dict[str, Any]):
    
    
    """Trains a model on the given data and saves it to the given model path.

    Args:
    --
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): val features.
        y_train (pd.DataFrame): Training target.
        y_test (pd.DataFrame): val target.
        parameters (dict): Parameters defined in parameters.yml.

    Returns:
    --
        models (dict): Dictionary of trained models.
        scores (pd.DataFrame): Dataframe of model scores.
    """
   
    # Identify categorical features
    categorical_features = X_train.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    logger.info(f"Categorical features found: {categorical_features}")
    
    # Create model dictionary
    models_dict = {
        'CatBoostRegressor': CatBoostRegressor(),
        'XGBRegressor': XGBRegressor(),
    }

    initial_results = {}   

    with open('conf/local/mlflow.yml') as f:
        experiment_name = yaml.load(f, Loader=yaml.loader.SafeLoader)['tracking']['experiment']['name']
        experiment_id = _get_or_create_experiment_id(experiment_name)
        logger.info(experiment_id)


    logger.info('Starting first step of model selection : Comparing between model types')
    
    # Prepare one-hot encoded data for XGBoost
    if categorical_features:
        # Create one-hot encoded version of data for XGBoost
        OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        OH_cols_train = pd.DataFrame(
            OH_encoder.fit_transform(X_train[categorical_features]),
            index=X_train.index,
            columns=OH_encoder.get_feature_names_out(categorical_features)
        )
        
        OH_cols_test = pd.DataFrame(
            OH_encoder.transform(X_test[categorical_features]),
            index=X_test.index,
            columns=OH_encoder.get_feature_names_out(categorical_features)
        )
        
        # Create versions of X_train and X_test with one-hot encoded features for XGBoost
        X_train_encoded = pd.concat([X_train.drop(categorical_features, axis=1), OH_cols_train], axis=1)
        X_test_encoded = pd.concat([X_test.drop(categorical_features, axis=1), OH_cols_test], axis=1)
        
        logger.info(f"Created one-hot encoded features for XGBoost. Original shape: {X_train.shape}, Encoded shape: {X_train_encoded.shape}")
    
    for model_name, model in models_dict.items():
        with mlflow.start_run(experiment_id=experiment_id,nested=True):
            mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)
            y_train_ravel = np.ravel(y_train)
            
            # For CatBoost, use original data and specify categorical_features
            if model_name == 'CatBoostRegressor':
                # Convert categorical feature indices to column indices (CatBoost requires numeric indices)
                cat_features_idx = [X_train.columns.get_loc(col) for col in categorical_features] if categorical_features else None
                model = CatBoostRegressor(cat_features=cat_features_idx, verbose=False)
                model.fit(X_train, y_train_ravel)
                y_pred = model.predict(X_test)
                logger.info(f"Trained CatBoost with categorical features: {cat_features_idx}")
                
            # For XGBoost, use the one-hot encoded data
            elif model_name == 'XGBRegressor':
                model.fit(X_train_encoded, y_train_ravel)
                y_pred = model.predict(X_test_encoded)
                logger.info(f"Trained XGBoost with one-hot encoded features")
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = sqrt(mse)  # Calculate RMSE
            r2 = r2_score(y_test, y_pred)
            initial_results[model_name] = -rmse  # Using negative RMSE for consistent comparison
            
            # Store the model and data format for later use
            if model_name == 'CatBoostRegressor':
                models_dict[model_name] = (model, 'original', cat_features_idx)
            else:
                models_dict[model_name] = (model, 'encoded', None)
            
            # Log model and metrics
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            
            run_id = mlflow.last_active_run().info.run_id
            logger.info(f"Logged model : {model_name} in run {run_id}, MSE: {mse}, RMSE: {rmse}, R²: {r2}")
      # Lower RMSE is better, but we store negative RMSE, so we use max
    best_model_name = max(initial_results, key=initial_results.get)
    best_model, data_format, cat_features_idx = models_dict[best_model_name]

    logger.info(f"Best model is {best_model_name} with RMSE: {-initial_results[best_model_name]}")
    logger.info('Starting second step of model selection : Hyperparameter tuning')
    
    # Perform hyperparameter tuning with GridSearchCV
    param_grid = parameters['hyperparameters'][best_model_name]
    
    with mlflow.start_run(experiment_id=experiment_id,nested=True):
        # Choose the right data format for GridSearchCV based on the best model
        if data_format == 'original':
            # For CatBoost, we need to create a new model instance with the categorical features
            if best_model_name == 'CatBoostRegressor':
                base_model = CatBoostRegressor(cat_features=cat_features_idx, verbose=False)
                X_train_grid = X_train
                X_test_grid = X_test
            else:
                base_model = best_model
                X_train_grid = X_train
                X_test_grid = X_test
        else:
            # For XGBoost, use the encoded data
            base_model = best_model
            X_train_grid = X_train_encoded
            X_test_grid = X_test_encoded
        
        # Use negative MSE for scoring since GridSearchCV maximizes the score
        gridsearch = GridSearchCV(base_model, param_grid, cv=2, scoring='neg_mean_squared_error', n_jobs=-1)
        gridsearch.fit(X_train_grid, np.ravel(y_train))
        best_model = gridsearch.best_estimator_
        
        # Log best hyperparameters
        mlflow.log_params(gridsearch.best_params_)
        
        # Calculate and log RMSE from the best score (which is negative MSE)
        best_rmse = sqrt(-gridsearch.best_score_)
        mlflow.log_metric("rmse", best_rmse)
        mlflow.log_metric("r2", r2_score(y_test, best_model.predict(X_test_grid)))


    logger.info(f"Hypertuned model best score: {best_rmse} (RMSE)")
    
    # Calculate final metrics on val set
    y_pred = best_model.predict(X_test_grid)
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"Final model metrics - MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R²: {r2}")

    # For regression, lower RMSE is better
    if champion_dict['val_rmse'] > rmse:
        logger.info(f"New champion model is {best_model_name} with RMSE: {rmse} vs {champion_dict['val_rmse']} ")
        return best_model
    else:
        logger.info(f"Champion model is still {champion_dict['regressor']} with RMSE: {champion_dict['val_rmse']} vs {rmse} ")
        return champion_model