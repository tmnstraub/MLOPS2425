"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.19.13
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


def na_col_to_unknown(df: pd.DataFrame, parameters: t.Dict[str, t.Any]) -> pd.DataFrame:
    # Remove index column if it exists
    df = remove_index_column(df)
   
    for col in parameters:
        # It's good practice to check if the column exists in the DataFrame
        if col in df.columns:
            df[col] = df[col].fillna('unknown')
        else:
            # You could add a log or a print statement here to warn about missing columns
            print(f"Warning: Column '{col}' not found in the DataFrame and was skipped.")

    return df
    
# create me a finciton which drops the row for a list of colums if the value is na
def drop_row(parameters: t.Dict, df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=parameters)

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

