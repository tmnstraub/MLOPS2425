import logging
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import os
import pickle


def feature_selection( X_train: pd.DataFrame , y_train: pd.DataFrame,  parameters: Dict[str, Any]):

    log = logging.getLogger(__name__)
    log.info(f"We start with: {len(X_train.columns)} columns")

    if parameters["feature_selection"] == "rfe":
        y_train = np.ravel(y_train)
        # open pickle file with regressors
        try:
            with open(os.path.join(os.getcwd(), 'data', '06_models', 'champion_model.pkl'), 'rb') as f:
                classifier = pickle.load(f)
        except:
            classifier = RandomForestClassifier(**parameters['baseline_model_params'])

        rfe = RFE(classifier) 
        rfe = rfe.fit(X_train, y_train)
        f = rfe.get_support(1) #the most important features
        X_cols = X_train.columns[f].tolist()

    log.info(f"Number of best columns is: {len(X_cols)}")
    
    return X_cols