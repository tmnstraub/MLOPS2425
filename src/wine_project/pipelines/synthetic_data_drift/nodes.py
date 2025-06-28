"""
This is a boilerplate pipeline 'synthetic_data_drift'
generated using Kedro 0.19.14
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def load_and_sample_data(
    wine_raw_data: pd.DataFrame, 
    n_samples: int = 30000, 
    random_state: int = 42
) -> pd.DataFrame:
    """
    Load wine raw data and sample n_samples rows with configurable random_state.
    Logs the shape and head of the data.
    
    Args:
        wine_raw_data (pd.DataFrame): The raw wine dataset passed by Kedro
        n_samples (int): Number of samples to take from the dataset (default: 1000)
        random_state (int): Random state for reproducible sampling (default: 42)
        
    Returns:
        pd.DataFrame: Sampled DataFrame with n_samples rows
    """
    logger.info(f"Original dataset shape: {wine_raw_data.shape}")
    logger.info(f"Original dataset head:\n{wine_raw_data.head()}")
    
    # If the dataset has fewer rows than n_samples, return the entire dataset
    if len(wine_raw_data) <= n_samples:
        logger.info(f"Dataset has {len(wine_raw_data)} rows, which is <= {n_samples}. Returning entire dataset.")
        sampled_data = wine_raw_data.copy()
    else:
        # Sample n_samples rows with the specified random_state
        sampled_data = wine_raw_data.sample(n=n_samples, random_state=random_state)
        logger.info(f"Sampled {n_samples} rows from dataset of {len(wine_raw_data)} total rows")
    
    logger.info(f"Sampled dataset shape: {sampled_data.shape}")
    logger.info(f"Sampled dataset head:\n{sampled_data.head()}")
    
    return sampled_data


def apply_points_drift(
    df: pd.DataFrame, 
    loc: float, 
    scale: float, 
    points_column: str,
    clip_values: bool = True
) -> pd.DataFrame:
    """
    Apply drift to the points column of a sampled dataframe.
    
    Args:
        df: Input dataframe
        loc: Mean of the normal distribution for drift
        scale: Standard deviation of the normal distribution for drift
        points_column: Name of the column to apply drift to
        clip_values: Whether to clip values to [0, 100] range (default: True)
        
    Returns:
        DataFrame with drift applied to the points column
    """
    logger.info(f"Applying points drift with loc={loc}, scale={scale}")
    
    # Create a copy to avoid modifying the original dataframe
    drifted_df = df.copy()
    
    # Log statistics before applying drift
    original_mean = drifted_df[points_column].mean()
    original_std = drifted_df[points_column].std()
    logger.info(f"Before drift - Mean: {original_mean:.2f}, Std: {original_std:.2f}")
    
    # Apply drift using normal distribution
    drift_values = np.random.normal(loc, scale, size=len(drifted_df))
    drifted_df[points_column] = drifted_df[points_column] + drift_values
    
    # Optional clipping to [0, 100] range
    if clip_values:
        drifted_df[points_column] = np.clip(drifted_df[points_column], 0, 100)
        logger.info("Values clipped to [0, 100] range")
    
    # Log statistics after applying drift
    drifted_mean = drifted_df[points_column].mean()
    drifted_std = drifted_df[points_column].std()
    logger.info(f"After drift - Mean: {drifted_mean:.2f}, Std: {drifted_std:.2f}")
    
    return drifted_df


def save_drifted_data(drifted_df: pd.DataFrame) -> pd.DataFrame:
    """
    Save the drifted dataframe to allow Kedro catalog to persist it.
    
    Args:
        drifted_df: The dataframe with drift applied
        
    Returns:
        The same dataframe for catalog persistence
    """
    logger.info(f"Saving drifted data with shape: {drifted_df.shape}")
    return drifted_df
