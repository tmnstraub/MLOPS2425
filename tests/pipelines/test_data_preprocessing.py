"""
This module contains an example test.

Tests should be placed in ``src/tests``, in modules that mirror your
project's structure, and in files named test_*.py. They are simply functions
named ``test_*`` which test a unit of logic.

To run the tests, run ``kedro test`` from the project root directory.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import os

from src.wine_project.pipelines.preprocessing_train.nodes import clean_data, feature_engineer

def test_clean_date_type():
    df = pd.read_csv("./tests/pipelines/sample/sample.csv") 
    df_transformed, describe_to_dict_verified  = clean_data(df)
    isinstance(describe_to_dict_verified, dict)

def test_clean_date_null():
    df = pd.read_csv("./tests/pipelines/sample/sample.csv") 
    df_transformed, describe_to_dict_verified = clean_data(df)
    assert [col for col in df_transformed.columns if df_transformed[col].isnull().any()] == []

def test_feature_engineering_month():
    df = pd.read_csv("./tests/pipelines/sample/sample.csv") 
    df_final, transformation = feature_engineer(df)
    assert(len(df_final["month"].unique().tolist()) <= 12)

