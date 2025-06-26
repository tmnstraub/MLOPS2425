
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
        X_test (pd.DataFrame): Test features.
        y_train (pd.DataFrame): Training target.
        y_test (pd.DataFrame): Test target.
        parameters (dict): Parameters defined in parameters.yml.

    Returns:
    --
        models (dict): Dictionary of trained models.
        scores (pd.DataFrame): Dataframe of model scores.
    """
   
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
    
    for model_name, model in models_dict.items():
        with mlflow.start_run(experiment_id=experiment_id,nested=True):
            mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)
            y_train = np.ravel(y_train)
            model.fit(X_train, y_train)            # For regression, lower MSE is better, so we store negative MSE
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = sqrt(mse)  # Calculate RMSE
            r2 = r2_score(y_test, y_pred)
            initial_results[model_name] = -rmse  # Using negative RMSE for consistent comparison
            
            # Log model and metrics
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            
            run_id = mlflow.last_active_run().info.run_id
            logger.info(f"Logged model : {model_name} in run {run_id}, MSE: {mse}, RMSE: {rmse}, R²: {r2}")
      # Lower RMSE is better, but we store negative RMSE, so we use max
    best_model_name = max(initial_results, key=initial_results.get)
    best_model = models_dict[best_model_name]

    logger.info(f"Best model is {best_model_name} with RMSE: {-initial_results[best_model_name]}")
    logger.info('Starting second step of model selection : Hyperparameter tuning')    # Perform hyperparameter tuning with GridSearchCV
    param_grid = parameters['hyperparameters'][best_model_name]
    with mlflow.start_run(experiment_id=experiment_id,nested=True):
        # Use negative MSE for scoring since GridSearchCV maximizes the score
        gridsearch = GridSearchCV(best_model, param_grid, cv=2, scoring='neg_mean_squared_error', n_jobs=-1)
        gridsearch.fit(X_train, y_train)
        best_model = gridsearch.best_estimator_
        
        # Log best hyperparameters
        mlflow.log_params(gridsearch.best_params_)
        
        # Calculate and log RMSE from the best score (which is negative MSE)
        best_rmse = sqrt(-gridsearch.best_score_)
        mlflow.log_metric("rmse", best_rmse)
        mlflow.log_metric("r2", r2_score(y_test, best_model.predict(X_test)))


    logger.info(f"Hypertuned model best score: {best_rmse} (RMSE)")
    
    # Calculate final metrics on test set
    y_pred = best_model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"Final model metrics - MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R²: {r2}")

    # For regression, lower RMSE is better
    if champion_dict['test_score'] > rmse:
        logger.info(f"New champion model is {best_model_name} with RMSE: {rmse} vs {champion_dict['test_score']} ")
        return best_model
    else:
        logger.info(f"Champion model is still {champion_dict['regressor']} with RMSE: {champion_dict['test_score']} vs {rmse} ")
        return champion_model