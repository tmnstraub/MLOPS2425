import pandas as pd
import logging
from typing import Dict, Tuple, Any
import numpy as np
import pickle
import yaml
import os
import warnings
warnings.filterwarnings("ignore", category=Warning)
import mlflow
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)

def model_train(X_train: pd.DataFrame, 
                X_val: pd.DataFrame, 
                y_train: pd.DataFrame, 
                y_val: pd.DataFrame,
                parameters: Dict[str, Any], best_columns):
    """Trains a model on the given data and saves it to the given model path.

    Args:
    --
        X_train (pd.DataFrame): Training features.
        X_val (pd.DataFrame): val features.
        y_train (pd.DataFrame): Training target.
        y_val (pd.DataFrame): val target.

    Returns:
    --
        model (pickle): Trained models.
        scores (json): Trained model metrics.
    """

    # enable autologging with error handling
    try:
        # Configure MLflow
        with open('conf/local/mlflow.yml') as f:
            mlflow_config = yaml.load(f, Loader=yaml.loader.SafeLoader)
            experiment_name = mlflow_config['tracking']['experiment']['name']
            tracking_uri = mlflow_config['server']['mlflow_tracking_uri']
        
        # Set the tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        
        # Get or create the experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
        
        logger.info(f'Starting model selection with MLflow tracking in experiment: {experiment_name}')
        mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)
        mlflow_enabled = True
    except Exception as e:
        logger.warning(f"MLflow setup failed: {e}. Continuing without MLflow tracking.")
        experiment_id = None
        mlflow_enabled = False

    # open pickle file with regressors
    try:
        with open(os.path.join(os.getcwd(), 'data', '06_models', 'champion_model.pkl'), 'rb') as f:
            loaded_obj = pickle.load(f)
            
            # Handle case where loaded object is a list or other container
            if isinstance(loaded_obj, list):
                logger.warning(f"Loaded model is a list with {len(loaded_obj)} items. Using a new CatBoost model.")
                regressor = CatBoostRegressor(**parameters['baseline_model_params'])
            elif isinstance(loaded_obj, dict):
                logger.warning("Loaded model is a dictionary. Using a new CatBoost model.")
                regressor = CatBoostRegressor(**parameters['baseline_model_params'])
            elif hasattr(loaded_obj, 'fit') and hasattr(loaded_obj, 'predict'):
                regressor = loaded_obj
                logger.info(f"Loaded model: {type(regressor).__name__}")
                
                if regressor.__class__.__name__ == 'CatBoostRegressor':
                    logger.info("Processing CatBoostRegressor parameters")
                    # Extract params from the classifier to use for the regressor
                    params = regressor.get_params()
                    # Remove classifier-specific parameters that don't apply to regressors
                    for param in ['loss_function', 'classes_count', 'class_weights']:
                        if param in params:
                            del params[param]
                    # Create a new regressor with the same parameters
                    regressor = CatBoostRegressor(**params)
            else:
                logger.warning(f"Loaded object is not usable as a model. Using a new CatBoost model.")
                regressor = CatBoostRegressor(**parameters['baseline_model_params'])
    except Exception as e:
        logger.warning(f"Error loading champion model: {str(e)}. Creating a new model.")
        regressor = CatBoostRegressor(**parameters['baseline_model_params'])

    results_dict = {}
    
    # Define a context manager that works with or without MLflow
    if mlflow_enabled:
        mlflow_context = mlflow.start_run(experiment_id=experiment_id, nested=True)
    else:
        from contextlib import nullcontext
        mlflow_context = nullcontext()
    
    # Use the appropriate context manager
    with mlflow_context:
        # Apply feature selection first
        if parameters["use_feature_selection"]:
            logger.info(f"Using feature selection in model train...")
            X_train = X_train[best_columns]
            X_val = X_val[best_columns]
                
        y_train = np.ravel(y_train)
        if isinstance(regressor, CatBoostRegressor):
            # CatBoost requires categorical features to be specified
            categorical_features = X_train.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
            logger.info(f"Categorical features found: {categorical_features}")
            # Calculate indices based on current X_train columns
            cat_features_idx = [X_train.columns.get_loc(col) for col in categorical_features] if categorical_features else None
            regressor.set_params(cat_features=cat_features_idx)

        model = regressor.fit(X_train, y_train)
        logger.info(f"Successfully trained {regressor.__class__.__name__} model on {X_train.shape[1]} features")
        
        # making predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        # evaluating model
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
        # saving results in dict
        results_dict['regressor'] = regressor.__class__.__name__
        results_dict['train_score'] = rmse_train
        results_dict['val_score'] = rmse_val
        
        # logging in mlflow if enabled
        if mlflow_enabled:
            try:
                run_id = mlflow.last_active_run().info.run_id
                logger.info(f"Logged train model in run {run_id}")
            except Exception as e:
                logger.warning(f"Failed to log to MLflow: {e}")
        
        logger.info(f"RMSE on validation set: {rmse_val}")



    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_train)

    

    shap.initjs()
    # calculate shap values. This is what we will plot.
    # For regression tasks, we don't need to specify a class index since there's only one output
    # (unlike classification where we had to use shap_values[:,:,1] for class 1)
    shap.summary_plot(shap_values, X_train, feature_names=X_train.columns, show=False)

    return model, X_train.columns , results_dict,plt