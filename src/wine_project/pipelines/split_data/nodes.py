"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


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


def split_random(df):
    # Remove index column if it exists
    df = remove_index_column(df)

    ref_data = df.sample(frac=0.8, random_state=200)
    ana_data = df.drop(ref_data.index)

    return ref_data, ana_data