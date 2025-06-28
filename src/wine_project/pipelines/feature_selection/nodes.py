import logging
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
import os
import pickle
from scipy.stats import chi2_contingency
import datetime  # Add this import for timestamp
from catboost import CatBoostRegressor


def feature_selection( X_train: pd.DataFrame , y_train: pd.DataFrame,  parameters: Dict[str, Any]):

    log = logging.getLogger(__name__)
    log.info(f"We start with: {len(X_train.columns)} columns")

    if parameters["feature_selection"] == "rfe":
        y_train = np.ravel(y_train)
        # open pickle file with regressors
        try:
            with open(os.path.join(os.getcwd(), 'data', '06_models', 'champion_model.pkl'), 'rb') as f:
                regressor = pickle.load(f)
        except:
            regressor = XGBRegressor(**parameters['baseline_model_params'])

        rfe = RFE(regressor) 
        rfe = rfe.fit(X_train, y_train)
        f = rfe.get_support(1) #the most important features
        X_cols = X_train.columns[f].tolist()
    
    if parameters["feature_selection"] == "tree_based":
        y_train = np.ravel(y_train)
        # Use RandomForestRegressor for feature importance
        forest = RandomForestRegressor(criterion="squared_error", max_depth=16)
        # Fit the forest to get feature importances
        forest.fit(X_train, y_train)
        
        # Use SelectFromModel with a threshold
        # Default threshold is 'mean', which selects features with importance > mean importance
        selector = SelectFromModel(forest, threshold='mean')
        selector.fit(X_train, y_train)
        
        # Get the selected features
        f = selector.get_support(1)  # The most important features
        X_cols = X_train.columns[f].tolist()
        
        # Log feature importances for transparency
        importances = forest.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_features = [(X_train.columns[i], importances[i]) for i in indices[:min(10, len(indices))]]
        log.info(f"Top features by importance: {top_features}")

    if parameters["feature_selection"] == "catboost":
        try:
            with open(os.path.join(os.getcwd(), 'data', '06_models', 'champion_model.pkl'), 'rb') as f:
                regressor = pickle.load(f)
        except:
            regressor = CatBoostRegressor(**parameters['baseline_model_params'])

        y_train = np.ravel(y_train)
        
        # Identify categorical features
        categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        cat_feature_indices = [X_train.columns.get_loc(col) for col in categorical_features]
        
        # Fit the model to get feature importances
        regressor.fit(X_train, y_train)
        
        # Get feature importances
        importances = regressor.get_feature_importance()
        
        # Sort features by importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Select top features based on importance threshold
        # You can adjust the threshold based on your needs
        importance_threshold = np.mean(importances)  # Use mean importance as threshold
        
        # Select features with importance above threshold
        selected_features = feature_importance[feature_importance['importance'] > importance_threshold]['feature'].tolist()
        
        # Log the selected features and their importance
        log.info(f"CatBoost feature selection - importance threshold: {importance_threshold:.6f}")
        log.info(f"Top 10 features by importance:")
        for i, row in feature_importance.head(10).iterrows():
            log.info(f"  {row['feature']}: {row['importance']:.6f}")
        
        X_cols = selected_features

    log.info(f"Number of best columns is: {len(X_cols)}")
    
    return X_cols

# def cramers_v(x: pd.Series, y: pd.Series) -> float:
#     """
#     Calculate Cramér's V correlation between two categorical variables.
    
#     Cramér's V is a measure of association between two categorical variables, 
#     giving a value between 0 (no association) and 1 (complete association).
    
#     Args:
#         x: First categorical variable
#         y: Second categorical variable
        
#     Returns:
#         float: Cramér's V correlation value
#     """
#     confusion_matrix = pd.crosstab(x, y)
#     chi2 = chi2_contingency(confusion_matrix)[0]
#     n = confusion_matrix.sum().sum()
#     phi2 = chi2 / n
#     r, k = confusion_matrix.shape
#     phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
#     rcorr = r - ((r - 1) ** 2) / (n - 1)
#     kcorr = k - ((k - 1) ** 2) / (n - 1)
    
#     # Handle division by zero
#     if min((kcorr - 1), (rcorr - 1)) == 0:
#         return 0.0
        
#     return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


# def feature_selection(X_train: pd.DataFrame, y_train: pd.DataFrame, parameters: Dict[str, Any] = None):
#     log = logging.getLogger(__name__)
#     log.info(f"We start with: {len(X_train.columns)} columns")
    
#     # Initialize parameters as empty dict if None or not a dict
#     if parameters is None or not isinstance(parameters, dict):
#         log.warning(f"Parameters is not a dictionary: {type(parameters)}. Using default values.")
#         parameters = {}
    
#     # Step 1: Separate numeric and categorical features
#     categorical_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
#     numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    
#     log.info(f"Found {len(numeric_cols)} numeric features and {len(categorical_cols)} categorical features")
    
#     # Step 2: Check for correlations in categorical features using Cramér's V
#     categorical_to_drop = []
#     if len(categorical_cols) > 1:
#         cramers_threshold = parameters.get("cramers_v_threshold", 0.7)
#         log.info(f"Checking categorical correlations using Cramér's V (threshold: {cramers_threshold})")
        
#         # Build correlation matrix for categorical features
#         for i in range(len(categorical_cols)):
#             for j in range(i+1, len(categorical_cols)):
#                 col1 = categorical_cols[i]
#                 col2 = categorical_cols[j]
#                 try:
#                     corr = cramers_v(X_train[col1], X_train[col2])
#                     if corr > cramers_threshold:
#                         log.info(f"High categorical correlation ({corr:.4f}) between {col1} and {col2}")
#                         # We'll keep the first column and drop the second
#                         categorical_to_drop.append(col2)
#                 except Exception as e:
#                     log.warning(f"Error calculating Cramér's V between {col1} and {col2}: {str(e)}")
    
#     # Remove duplicates from the drop list
#     categorical_to_drop = list(set(categorical_to_drop))
#     if categorical_to_drop:
#         log.info(f"Removing {len(categorical_to_drop)} highly correlated categorical features: {categorical_to_drop}")
        
#     # Get the remaining categorical columns
#     categorical_cols = [col for col in categorical_cols if col not in categorical_to_drop]
    
#     # Step 3: Combine the filtered features
#     filtered_cols = numeric_cols + categorical_cols
#     X_train_filtered = X_train[filtered_cols]
#     log.info(f"After categorical correlation filtering: {len(filtered_cols)} columns")
    
#     # Step 4: Apply RFE if selected
#     X_cols = filtered_cols  # Default to filtered columns
#     if parameters.get("feature_selection", "").lower() == "rfe":
#         y_train = np.ravel(y_train)
        
#         # Load one-hot encoded data from the specified path
#         one_hot_path = os.path.join(os.getcwd(), 'data', '05_train_val_split', 'X_train_one_hot.csv')
#         log.info(f"Loading one-hot encoded data from: {one_hot_path}")
        
#         # Load one-hot encoded data - if this fails, the error will propagate
#         X_train_one_hot = pd.read_csv(one_hot_path)
#         log.info(f"Successfully loaded one-hot encoded data with shape: {X_train_one_hot.shape}")
        
#         # Ensure indices match between X_train_one_hot and y_train
#         if len(X_train_one_hot) != len(y_train):
#             log.warning(f"Shape mismatch: X_train_one_hot has {len(X_train_one_hot)} rows, " 
#                        f"but y_train has {len(y_train)} rows. Will attempt to align.")
            
#             # Try to align indices if possible
#             if hasattr(y_train, 'index') and hasattr(X_train_one_hot, 'index'):
#                 common_indices = X_train_one_hot.index.intersection(y_train.index)
#                 if len(common_indices) > 0:
#                     X_train_one_hot = X_train_one_hot.loc[common_indices]
#                     y_train = y_train.loc[common_indices]
#                     log.info(f"Aligned data using common indices. New shapes: X={X_train_one_hot.shape}, y={y_train.shape}")
        
#         # Open pickle file with regressors
#         try:
#             with open(os.path.join(os.getcwd(), 'data', '06_models', 'champion_model.pkl'), 'rb') as f:
#                 classifier = pickle.load(f)
#         except Exception as e:
#             log.warning(f"Could not load champion model: {str(e)}. Using default classifier.")
#             classifier = RandomForestClassifier(**parameters.get('baseline_model_params', {'n_estimators': 100}))

#         # Apply RFE
#         n_features_to_select = parameters.get("n_features_to_select", None)
        
#         # Ensure we have a valid number of features to select
#         if n_features_to_select is not None:
#             try:
#                 n_features_to_select = int(n_features_to_select)
#                 if n_features_to_select > len(X_train_one_hot.columns):
#                     log.warning(f"n_features_to_select ({n_features_to_select}) is greater than number of features "
#                                f"({len(X_train_one_hot.columns)}). Using all features.")
#                     n_features_to_select = None
#             except ValueError:
#                 log.warning(f"Invalid n_features_to_select value: {n_features_to_select}. Using default.")
#                 n_features_to_select = None
        
#         # Run RFE on the one-hot encoded data
#         rfe = RFE(classifier, n_features_to_select=n_features_to_select)
#         log.info("Fitting RFE with one-hot encoded data...")
#         rfe = rfe.fit(X_train_one_hot, y_train)
        
#         # Get selected features
#         selected_indices = rfe.get_support(1)  # The most important features
#         X_cols = X_train_one_hot.columns[selected_indices].tolist()
#         log.info(f"RFE selected {len(X_cols)} features from one-hot encoded data")
        
#         # Save RFE results for analysis
#         output_dir = os.path.join(os.getcwd(), 'data', '04_feature')
#         os.makedirs(output_dir, exist_ok=True)
        
#         # Define timestamp for file naming
#         timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
#         # Save feature ranking and selection status
#         feature_ranking = pd.DataFrame({
#             'feature': X_train_one_hot.columns,
#             'selected': rfe.support_,
#             'rank': rfe.ranking_
#         })
#         feature_ranking = feature_ranking.sort_values('rank')
        
#         # Save feature ranking
#         feature_ranking_path = os.path.join(output_dir, f"feature_ranking_{timestamp}.csv")
#         feature_ranking.to_csv(feature_ranking_path, index=False)
        
#         # Save the selected features separately
#         selected_features_df = pd.DataFrame({'feature': X_cols})
#         selected_features_path = os.path.join(output_dir, f"selected_features_{timestamp}.csv")
#         selected_features_df.to_csv(selected_features_path, index=False)
        
#         log.info(f"Feature selection results saved to {output_dir}")
#         log.info(f"- Feature ranking: {feature_ranking_path}")
#         log.info(f"- Selected features: {selected_features_path}")

#     log.info(f"Number of best columns is: {len(X_cols)}")
    
#     return X_cols