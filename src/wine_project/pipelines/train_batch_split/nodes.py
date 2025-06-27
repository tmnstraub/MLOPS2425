"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def split_random(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the input dataframe into a training set and a batch set.
    Training set (80%) will be further processed and split into train/validation sets.
    Batch set (20%) will be treated as unseen data for final evaluation.
    
    Args:
        df: Input dataframe to split
        
    Returns:
        Tuple containing (train_data, batch_data)
    """
    train_data = df.sample(frac=0.8, random_state=200)
    batch_data = df.drop(train_data.index)

    return train_data, batch_data