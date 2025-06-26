"""
This pipeline handles preprocessing for the batch dataset (treated as unseen data)
"""

import pandas as pd 
from great_expectations.core import ExpectationSuite, ExpectationConfiguration
import typing as t

def remove_index_column(data):
    """
    Remove index column from the DataFrame.
    
    Args:
        data: Pandas DataFrame that might have an index or df_index column
        
    Returns:
        DataFrame with index column removed
    """
    # Check for 'df_index' column
    if 'df_index' in data.columns:
        data = data.drop(columns=['df_index'])
    
    # Check for 'index' column
    if 'index' in data.columns:
        data = data.drop(columns=['index'])
    
    return data

def drop_col(parameters: t.Dict, df: pd.DataFrame) -> pd.DataFrame:
    # Remove index column if it exists
    df = remove_index_column(df)
    return df.drop(columns = parameters)

def na_col_to_unknown(df: pd.DataFrame) -> pd.DataFrame:
    # Remove index column if it exists
    df = remove_index_column(df)

    categorical_columns = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
   
    df[categorical_columns] = df[categorical_columns].fillna('unknown')

    return df
    
# def drop_row(df: pd.DataFrame) -> pd.DataFrame:
#     return df.dropna()

def drop_zero_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where the price is zero.
    
    Args:
        df: Pandas DataFrame with a price column
        
    Returns:
        DataFrame with zero price rows removed
    """
    # Count rows before filtering
    rows_before = df.shape[0]
    
    # Filter out records with price value of 0
    df = df[df['price'] != 0]
    
    # Count rows after filtering
    rows_after = df.shape[0]
    rows_removed = rows_before - rows_after
    
    # Reset index to make sure we have consecutive indices
    df = df.reset_index(drop=True)
    
    print(f"Removed {rows_removed} records with zero prices out of {rows_before} total records")
    
    return df

def preprocess_batch_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main preprocessing function for the batch data. Uses the same preprocessing 
    steps as the training data to ensure consistency.
    
    Args:
        df: Batch data DataFrame
    
    Returns:
        Preprocessed batch DataFrame
    """
    # Make a copy to avoid modifying the input
    processed_df = df.copy()
    
    # Apply preprocessing steps in sequence
    processed_df = remove_index_column(processed_df)
    
    # Apply common preprocessing steps
    processed_df = drop_zero_prices(processed_df)

    #Apply the unknown value replacement
    processed_df = na_col_to_unknown(processed_df)
    
    return processed_df
    
