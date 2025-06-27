
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
from sklearn.metrics import root_mean_squared_error
from catboost import CatBoostClassifier
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

    # enable autologging
    with open('conf/local/mlflow.yml') as f:
        experiment_name = yaml.load(f, Loader=yaml.loader.SafeLoader)['tracking']['experiment']['name']
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    logger.info('Starting first step of model selection : Comparing between modes types')
    mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)

    # open pickle file with regressors
    try:
        with open(os.path.join(os.getcwd(), 'data', '06_models', 'champion_model.pkl'), 'rb') as f:
            classifier = pickle.load(f)
    except:
        classifier = CatBoostClassifier(**parameters['baseline_model_params'])

    results_dict = {}
    with mlflow.start_run(experiment_id=experiment_id, nested=True):
        if parameters["use_feature_selection"]:
            logger.info(f"Using feature selection in model train...")
            X_train = X_train[best_columns]
            X_val = X_val[best_columns]
        y_train = np.ravel(y_train)
        model = classifier.fit(X_train, y_train)
        # making predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        # evaluating model
        rmse_train = root_mean_squared_error(y_train, y_train_pred)
        rmse_val = root_mean_squared_error(y_val, y_val_pred)
        # saving results in dict
        results_dict['classifier'] = classifier.__class__.__name__
        results_dict['train_score'] = rmse_train
        results_dict['val_score'] = rmse_val
        # logging in mlflow
        run_id = mlflow.last_active_run().info.run_id
        logger.info(f"Logged train model in run {run_id}")
        logger.info(f"rmseuracy is {rmse_val}")



    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_train)

    

    shap.initjs()
    # calculate shap values. This is what we will plot.
    # For regression tasks, we don't need to specify a class index since there's only one output
    # (unlike classification where we had to use shap_values[:,:,1] for class 1)
    shap.summary_plot(shap_values, X_train, feature_names=X_train.columns, show=False)

    return model, X_train.columns , results_dict,plt