"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

import great_expectations as ge
from great_expectations.core import ExpectationSuite, ExpectationConfiguration

import re
import logging
from pathlib import Path

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

# Add this code block right before import hopsworks (around line 88)
import hsfs
import sys

# Create mock module to handle missing hopsworks_udf
class MockHopsworksUdf:
    def __init__(self):
        self.udf = None

# Add the missing module to hsfs
if not hasattr(hsfs, 'hopsworks_udf'):
    hsfs.hopsworks_udf = MockHopsworksUdf()

conf_path = str(Path('') / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)
credentials = conf_loader["credentials"]

logger = logging.getLogger(__name__)

conf_path = str(Path('') / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)
credentials = conf_loader["credentials"]

logger = logging.getLogger(__name__)

def build_expectation_suite(expectation_suite_name: str, feature_group: str, columns: list = None) -> ExpectationSuite:
    """
    Builder used to retrieve an instance of the validation expectation suite.
    
    Args:
        expectation_suite_name (str): A dictionary with the feature group name and the respective version.
        feature_group (str): Feature group used to construct the expectations.
        columns (list, optional): List of column names available in the feature group
             
    Returns:
        ExpectationSuite: A dictionary containing all the expectations for this particular feature group.
    """
    
    expectation_suite_wine = ExpectationSuite(
        expectation_suite_name=expectation_suite_name
    )
    
    # Ensure columns is at least an empty list
    if columns is None:
        columns = []
    
    # numerical features
    if feature_group == 'numerical_features':
        if 'points' in columns:
            expectation_suite_wine.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_in_type_list",
                    kwargs={"column": "points", "type_list": ["int64"]},
                )
            )

    if feature_group == 'categorical_features':
        categorical_cols = ["country", "province", "region_1", "taster_name", "variety"]
        for col in categorical_cols:
            # Only add expectations for columns that exist in this feature group
            if col in columns:
                expectation_suite_wine.add_expectation(
                    ExpectationConfiguration(
                        expectation_type="expect_column_values_to_be_of_type",
                        kwargs={"column": col, "type_": "str"},
                    )
                )

    if feature_group == 'target':
        if 'price' in columns:
            expectation_suite_wine.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_in_type_list",
                    kwargs={"column": "price", "type_list": ["float64"]},
                )
            )
    
    return expectation_suite_wine

import hopsworks

def to_feature_store(
    data: pd.DataFrame,
    group_name: str,
    feature_group_version: int,
    description: str,
    group_description: dict,
    validation_expectation_suite: ExpectationSuite,
    credentials_input: dict
):
    """
    This function takes in a pandas DataFrame and a validation expectation suite,
    performs validation on the data using the suite, and then saves the data to a
    feature store in the feature store.

    Args:
        data (pd.DataFrame): Dataframe with the data to be stored
        group_name (str): Name of the feature group.
        feature_group_version (int): Version of the feature group.
        description (str): Description for the feature group.
        group_description (dict): Description of each feature of the feature group. 
        validation_expectation_suite (ExpectationSuite): group of expectations to check data.
        SETTINGS (dict): Dictionary with the settings definitions to connect to the project.
        
    Returns:
        A dictionary with the feature view version, feature view name and training dataset feature version.
    """
    # Check if required credentials are present
    if not credentials_input or not isinstance(credentials_input, dict):
        raise ValueError("Feature store credentials not provided or invalid")
    
    required_keys = ["FS_API_KEY", "FS_PROJECT_NAME"]
    missing_keys = [key for key in required_keys if key not in credentials_input]
    if missing_keys:
        raise ValueError(f"Missing required credential keys: {', '.join(missing_keys)}")
    
    # Connect to feature store.
    try:
        project = hopsworks.login(
            api_key_value=credentials_input["FS_API_KEY"], 
            project=credentials_input["FS_PROJECT_NAME"]
        )
        feature_store = project.get_feature_store()
        logger.info(f"Successfully connected to Hopsworks feature store")
    except Exception as e:
        logger.error(f"Error connecting to feature store: {e}")
        raise e

    # Clean column names to meet Hopsworks requirements
    # Feature names can only contain lowercase letters, numbers and underscores
    # Must start with a letter and cannot be longer than 63 characters
    import re
    data = data.copy()
    
    # Use row_id instead of index
    data['row_id'] = range(len(data))
    
    # Ensure datetime column exists and is properly formatted
    if 'datetime' not in data.columns:
        data['datetime'] = pd.Timestamp.now()
    else:
        # Make sure datetime is properly formatted as timestamp
        data['datetime'] = pd.to_datetime(data['datetime'])
    
    clean_columns = {}
    for col in data.columns:
        # Convert to lowercase
        clean_col = str(col).lower()
        # Replace non-alphanumeric characters with underscore
        clean_col = re.sub(r'[^a-z0-9_]', '_', clean_col)
        # Ensure it starts with a letter
        if not clean_col[0].isalpha():
            clean_col = 'f_' + clean_col
        # Ensure it's not too long
        if len(clean_col) > 63:
            clean_col = clean_col[:63]
        clean_columns[col] = clean_col
    
    # Rename columns
    data = data.rename(columns=clean_columns)

    # Create feature group with explicit schema
    schema = []
    for col in data.columns:
        col_type = None
        if col == 'row_id':
            col_type = 'int'
        elif col == 'datetime':
            col_type = 'timestamp'
        elif pd.api.types.is_integer_dtype(data[col]):
            col_type = 'int'
            data[col] = data[col].astype('int32')
        elif pd.api.types.is_float_dtype(data[col]):
            col_type = 'double'
            data[col] = data[col].astype('float32')
        else:
            col_type = 'string'
            data[col] = data[col].astype(str)
        
        schema.append({'name': col, 'type': col_type})
    
    logger.info(f"Feature group schema: {schema}")
    
    # Create feature group with explicit schema
    try:
        # First check if feature group exists
        try:
            # Try to get the existing feature group
            object_feature_group = feature_store.get_feature_group(
                name=group_name,
                version=feature_group_version
            )
            logger.info(f"Feature group {group_name} already exists, using existing group")
            
            # For existing feature groups, we need to delete it first to avoid schema conflicts
            logger.info(f"Deleting existing feature group to avoid schema conflicts")
            object_feature_group.delete()
            logger.info(f"Successfully deleted existing feature group")
            
            # Create a new feature group with the updated schema
            object_feature_group = feature_store.create_feature_group(
                name=group_name,
                version=feature_group_version,
                description=description,
                primary_key=["row_id"],
                event_time="datetime",
                online_enabled=False,
                expectation_suite=validation_expectation_suite,
                statistics_config={
                    "enabled": True,
                    "histograms": True,
                    "correlations": True,
                }
            )
            logger.info(f"Recreated feature group with updated schema: {group_name}")
            
        except Exception as get_error:
            logger.info(f"Feature group does not exist or could not be retrieved: {get_error}")
            # Create new feature group with schema
            object_feature_group = feature_store.create_feature_group(
                name=group_name,
                version=feature_group_version,
                description=description,
                primary_key=["row_id"],
                event_time="datetime",
                online_enabled=False,
                expectation_suite=validation_expectation_suite,
                statistics_config={
                    "enabled": True,
                    "histograms": True,
                    "correlations": True,
                }
            )
            logger.info(f"Created new feature group: {group_name}")
    except Exception as e:
        logger.error(f"Error creating feature group: {e}")
        # Return None instead of raising exception so we can continue with the rest of the pipeline
        return None
    
    # Upload data with detailed error handling
    try:
        # Skip upload if feature group creation failed
        if object_feature_group is None:
            logger.warning(f"Skipping data upload because feature group creation failed")
            return None
            
        # Convert data types to be compatible with feature store
        for col in data.columns:
            if col == 'datetime':
                # Ensure datetime is correctly formatted
                data[col] = pd.to_datetime(data[col])
            elif pd.api.types.is_integer_dtype(data[col]):
                data[col] = data[col].astype('int32')
            elif pd.api.types.is_float_dtype(data[col]):
                data[col] = data[col].astype('float32')
            else:
                data[col] = data[col].astype(str)
        
        # Log data sample and types for debugging
        logger.info(f"Data sample: {data.head(2)}")
        logger.info(f"Data columns: {data.columns.tolist()}")
        logger.info(f"Data types: {data.dtypes}")
        
        # Insert data
        object_feature_group.insert(
            features=data,
            overwrite=True,
            write_options={
                "wait_for_job": True,
            },
        )
        logger.info(f"Successfully uploaded data to feature group: {group_name}")
        
        return object_feature_group
        
    except Exception as e:
        logger.error(f"Error inserting data into feature store: {e}")

        return None


def ingestion(
    df1: pd.DataFrame,
    parameters: Dict[str, Any]):

    """
    This function takes in a pandas DataFrame and a validation expectation suite,
    performs validation on the data using the suite, and then saves the data to a
    feature store in the feature store.

    Args:
        data (pd.DataFrame): Dataframe with the data to be stored
        group_name (str): Name of the feature group.
        feature_group_version (int): Version of the feature group.
        description (str): Description for the feature group.
        group_description (dict): Description of each feature of the feature group. 
        validation_expectation_suite (ExpectationSuite): group of expectations to check data.
        SETTINGS (dict): Dictionary with the settings definitions to connect to the project.
        
    Returns:
       A DataFrame with the processed data
    
    """
    # Load credentials if using feature store
    credentials = {}
    if parameters.get("to_feature_store", False):
        try:
            import yaml
            import os
            
            # Direct path to credentials file
            cred_path = os.path.join(settings.CONF_SOURCE, "local", "credentials.yml")
            logger.info(f"Attempting to load credentials from: {cred_path}")
            
            if os.path.exists(cred_path):
                with open(cred_path, 'r') as file:
                    credentials = yaml.safe_load(file)
                logger.info(f"Successfully loaded credentials. Feature store access enabled.")
                logger.info(f"Credential file keys: {list(credentials.keys())}")
            else:
                logger.warning(f"Credentials file not found at {cred_path}")
                parameters["to_feature_store"] = False
        except Exception as e:
            logger.warning(f"Failed to load credentials: {e}")
            # Disable feature store if credentials can't be loaded
            parameters["to_feature_store"] = False

    common_columns= []
    for i in df1.columns.tolist():
        if i in df1.columns.tolist():
            common_columns.append(i)
    
    assert len(common_columns)>0, "Wrong data collected"

    df_clean = df1.drop(columns=["region_2", "designation", "description", "title", "winery", "taster_twitter_handle"]) 

    df_clean = df_clean.drop_duplicates()
    df_clean = df_clean.dropna(how='all')

    logger.info(f"The dataset contains {len(df_clean.columns)} columns.")

    # Add filter for price greater than 0
    if parameters["target_column"] == "price":
        # Log before filtering
        initial_count = len(df_clean)
        logger.info(f"Initial dataset size before price filtering: {initial_count}")
        
        # Filter to keep only records with price > 0
        df_clean = df_clean[df_clean["price"] > 0]
        
        # Log after filtering
        filtered_count = len(df_clean)
        removed_count = initial_count - filtered_count
        logger.info(f"Removed {removed_count} records with price <= 0")
        logger.info(f"Dataset size after price filtering: {filtered_count}")

    # Check for unnamed columns and rename them
    unnamed_cols = [col for col in df_clean.columns if 'unnamed' in col.lower() or 'Unnamed' in col]
    for col in unnamed_cols:
        df_clean = df_clean.drop(columns=[col])
    
    # Reset index to ensure we have a clean index 
    df_clean = df_clean.reset_index(drop=True)
    
    # We're no longer adding an explicit index column
    
    # Remove any existing index or df_index columns if they exist
    if "df_index" in df_clean.columns:
        logger.info("Found df_index column, dropping it")
        df_clean = df_clean.drop(columns=["df_index"])
    
    if "index" in df_clean.columns:
        logger.info("Found index column, dropping it")
        df_clean = df_clean.drop(columns=["index"])
    
    # Create a proper datetime column
    df_clean["datetime"] = pd.to_datetime("2025-01-01")

    # Separate numerical and categorical features
    numerical_features = df_clean.select_dtypes(exclude=['object','string','category']).columns.tolist()
    # Make sure to exclude index and datetime from numerical features
    if 'index' in numerical_features:
        numerical_features.remove('index')
    if 'datetime' in numerical_features:
        numerical_features.remove('datetime')
        
    categorical_features = df_clean.select_dtypes(include=['object','string','category']).columns.tolist()
    # Exclude the target column and datetime from categorical features
    if parameters["target_column"] in categorical_features:
        categorical_features.remove(parameters["target_column"])
    if 'datetime' in categorical_features:
        categorical_features.remove('datetime')

    # Build validation expectations
    validation_expectation_suite_numerical = build_expectation_suite(
        "numerical_expectations", 
        "numerical_features", 
        numerical_features
    )
    
    validation_expectation_suite_categorical = build_expectation_suite(
        "categorical_expectations", 
        "categorical_features", 
        categorical_features
    )
    
    validation_expectation_suite_target = build_expectation_suite(
        "target_expectations", 
        "target", 
        [parameters["target_column"]]
    )

    numerical_feature_descriptions = []
    for col in numerical_features:
        numerical_feature_descriptions.append({
            "name": col,
            "description": f"Numerical feature: {col}"
        })
        
    categorical_feature_descriptions = []
    for col in categorical_features:
        categorical_feature_descriptions.append({
            "name": col,
            "description": f"Categorical feature: {col}"
        })
        
    target_feature_descriptions = [{
        "name": parameters["target_column"],
        "description": f"Target variable: {parameters['target_column']}"
    }]
    
    # Convert data types appropriately for feature store
    for col in df_clean.columns:
        if col == 'datetime':
            # Keep datetime as datetime
            pass
        elif col == 'index':
            # Keep index as integer
            df_clean[col] = df_clean[col].astype('int32')
        elif col in numerical_features:
            # Handle numeric features
            if pd.api.types.is_integer_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(0).astype('int32')
            else:
                df_clean[col] = df_clean[col].fillna(0.0).astype('float32')
        else:
            # Convert all categorical features to string
            df_clean[col] = df_clean[col].fillna('').astype(str)
    
    # Create subsets with required columns
    # Debug numerical features
    logger.info(f"All numerical features before cleanup: {numerical_features}")
    
    # Check for duplicates in column names
    df_cols = df_clean.columns.tolist()
    duplicate_cols = [col for col in df_cols if df_cols.count(col) > 1]
    if duplicate_cols:
        logger.warning(f"Duplicate columns found: {duplicate_cols}")
        # Remove duplicates from DataFrame
        df_clean = df_clean.loc[:, ~df_clean.columns.duplicated()]
        # Update numerical features
        numerical_features = df_clean.select_dtypes(exclude=['object','string','category']).columns.tolist()
        if 'index' in numerical_features:
            numerical_features.remove('index')
        if 'datetime' in numerical_features:
            numerical_features.remove('datetime')
        logger.info(f"Numerical features after removing duplicates: {numerical_features}")
    
    # Ensure we have unique columns in numerical features
    # Use a list instead of set to maintain column order, but ensure no duplicates
    unique_numerical_features = []
    for col in numerical_features:
        if col not in unique_numerical_features:
            unique_numerical_features.append(col)
    
    df_full_numeric = df_clean[["datetime"] + unique_numerical_features].copy()
    logger.info(f"Numerical features columns in df_full_numeric: {df_full_numeric.columns.tolist()}")
    logger.info(f"Numerical features dtypes: {df_full_numeric.dtypes}")
    
    df_full_categorical = df_clean[["datetime"] + categorical_features].copy()
    df_full_target = df_clean[["datetime"] + [parameters["target_column"]]].copy()

    if parameters.get("to_feature_store", False):
        try:
            # Add more detailed logging to debug credential issues
            logger.info(f"Credentials dictionary keys: {list(credentials.keys() if credentials else [])}")
            
            # Check if credentials has a feature_store key
            if "feature_store" in credentials:
                fs_credentials = credentials["feature_store"]
                logger.info("Using feature_store section from credentials")
            else:
                fs_credentials = credentials
                logger.info("Using entire credentials dictionary for feature store")
            
            # Debug the structure of fs_credentials
            logger.info(f"Feature store credentials keys: {list(fs_credentials.keys() if fs_credentials else [])}")
            
            # Try with numerical features
            try:
                # Ensure we don't have duplicate columns before calling feature store
                if len(df_full_numeric.columns) != len(set(df_full_numeric.columns)):
                    logger.error(f"Duplicate columns found in numerical features, cannot proceed with feature store upload.")
                    logger.info(f"Columns with potential duplicates: {df_full_numeric.columns.tolist()}")
                    # Get the duplicates for debugging
                    dup_cols = [col for col in df_full_numeric.columns.tolist() 
                               if df_full_numeric.columns.tolist().count(col) > 1]
                    logger.info(f"Duplicate columns: {dup_cols}")
                    # Try to fix by dropping duplicates
                    df_full_numeric_fixed = df_full_numeric.loc[:, ~df_full_numeric.columns.duplicated()]
                    logger.info(f"Fixed columns after dropping duplicates: {df_full_numeric_fixed.columns.tolist()}")
                    
                    # Use the fixed dataframe for feature store
                    object_fs_numerical_features = to_feature_store(
                        df_full_numeric_fixed, "numerical_features",
                        1, "Numerical Features",
                        numerical_feature_descriptions,
                        validation_expectation_suite_numerical,
                        fs_credentials
                    )
                else:
                    object_fs_numerical_features = to_feature_store(
                        df_full_numeric, "numerical_features",
                        1, "Numerical Features",
                        numerical_feature_descriptions,
                        validation_expectation_suite_numerical,
                        fs_credentials
                    )
                    
                if object_fs_numerical_features:
                    logger.info("Successfully uploaded numerical features")
            except Exception as e:
                logger.error(f"Error uploading numerical features: {e}")
                # Continue with other feature groups
            
            # Try with categorical features
            try:
                object_fs_categorical_features = to_feature_store(
                    df_full_categorical,"categorical_features",
                    1,"Categorical Features",
                    categorical_feature_descriptions,
                    validation_expectation_suite_categorical,
                    fs_credentials
                )
                logger.info("Successfully uploaded categorical features")
            except Exception as e:
                logger.error(f"Error uploading categorical features: {e}")
            
            # Try with target features
            try:
                object_fs_target_features = to_feature_store(
                    df_full_target,"target_features",
                    1,"Target Features",
                    target_feature_descriptions,
                    validation_expectation_suite_target,
                    fs_credentials
                )
                logger.info("Successfully uploaded target features")
            except Exception as e:
                logger.error(f"Error uploading target features: {e}")
                
        except Exception as e:
            logger.error(f"Error while uploading to feature store: {e}")
            logger.info("Continuing with local processing only.")

    return df_clean