import pandas as pd
import numpy as np
import pycountry_convert as pc
from sklearn.preprocessing import OneHotEncoder

# Import the relevant functions from train nodes to ensure consistency
from wine_project.pipelines.feature_engineering_train.nodes import (
    remove_index_column,
    classify_points,
    classify_wine_type_main,
    classify_wine_subtype,
    get_continent
)

def engineer_batch_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for the batch dataset.
    
    Args:
        df: Batch dataset
        
    Returns:
        DataFrame with engineered features
    """
    # Remove index column if it exists
    df = remove_index_column(df)
    df = df.copy()

    # Points category
    df['points_category'] = df['points'].apply(classify_points)

    # Blend flag - ensure the variety column exists and is string type
    if 'variety' in df.columns:
        # Convert to string first to handle any non-string values
        df['variety'] = df['variety'].astype(str)
        # Create is_blend feature
        df['is_blend'] = df['variety'].str.contains("blend", case=False)
    else:
        # Create a default is_blend column if variety doesn't exist
        df['is_blend'] = False

    # Wine classification
    df['wine_type_main'] = df['variety'].apply(classify_wine_type_main)
    df['wine_subtype'] = df['variety'].apply(classify_wine_subtype)
    
    # Continent classification (standardize US/UK first)
    df['country_standardized'] = df['country'].replace({
        'US': 'United States',
        'England': 'United Kingdom'
    })
    df['continent'] = df['country_standardized'].apply(get_continent)
    
    return df

def select_batch_features(df: pd.DataFrame, features_to_drop: list = None) -> pd.DataFrame:
    """
    Select features from the engineered batch dataset by dropping specified columns.
    
    Args:
        df: Feature-engineered batch dataset
        features_to_drop: List of column names to exclude from the dataset
        
    Returns:
        DataFrame with selected features
    """
    # Create a copy of the dataframe
    df_selected = df.copy()
    
    # If features_to_drop is not provided or empty, return the original dataframe
    if not features_to_drop:
        print("No features specified to drop for batch data. Keeping all features.")
        return df_selected
    
    # Print the features that will be dropped for debugging
    print(f"Features specified to drop from batch data: {features_to_drop}")
    
    # Filter the features_to_drop list to only include columns that actually exist in the dataframe
    valid_features_to_drop = [col for col in features_to_drop if col in df_selected.columns]
    
    # Check if any specified features don't exist in the dataframe
    if len(valid_features_to_drop) < len(features_to_drop):
        missing_features = set(features_to_drop) - set(valid_features_to_drop)
        print(f"Warning: The following features don't exist in the batch dataframe: {missing_features}")
    
    # Drop the specified features
    if valid_features_to_drop:
        print(f"Dropping {len(valid_features_to_drop)} features from batch data: {valid_features_to_drop}")
        df_selected = df_selected.drop(columns=valid_features_to_drop)
    
    return df_selected

def create_one_hot_encoded_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create one-hot encoded features from categorical columns using scikit-learn's OneHotEncoder.
    
    Args:
        df: DataFrame with categorical columns
        
    Returns:
        DataFrame with one-hot encoded features
    """
    # Create a copy of the dataframe to avoid modifying the original
    df_encoded = df.copy()
    
    # List of categorical columns to one-hot encode
    categorical_columns = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()

    # Initialize the OneHotEncoder
    # Setting handle_unknown='ignore' to handle categories not seen during fit
    # drop='first' to avoid multicollinearity
    encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    
    # Fit and transform the categorical columns
    if categorical_columns:  # Only proceed if there are categorical columns
        encoded_array = encoder.fit_transform(df_encoded[categorical_columns])
        
        # Get feature names from the encoder
        feature_names = encoder.get_feature_names_out(categorical_columns)
        
        # Create a dataframe with the encoded values
        encoded_df = pd.DataFrame(
            encoded_array,
            columns=feature_names,
            index=df_encoded.index
        )
        
        # Concatenate the original dataframe with the encoded columns
        df_encoded = pd.concat([df_encoded.drop(columns=categorical_columns), encoded_df], axis=1)
    
    # Check if 'is_blend' column exists before attempting to use it
    if 'is_blend' in df_encoded.columns:
        # One-hot encode the boolean column, handling NaN values
        # Fill NaN values with False (0) before converting to int
        df_encoded['is_blend_True'] = df_encoded['is_blend'].fillna(False).astype(int)
        
        # Drop the original is_blend column
        df_encoded = df_encoded.drop(columns=['is_blend'])
    
    return df_encoded
