"""
This is a boilerplate pipeline 'data_drift'
generated using Kedro 0.19.13
"""


# In src/<your_package>/pipelines/data_drift/nodes.py

import pandas as pd
import numpy as np
import nannyml as nml
from typing import Dict

# --- Helper Function ---
def _add_timestamp_from_index(df: pd.DataFrame, ts_col_name: str) -> pd.DataFrame:
    """
    Takes a DataFrame, creates a copy, and adds a new column 
    named `ts_col_name` from the DataFrame's index.
    """
    df_copy = df.copy()
    # reset_index() converts the index into a column named 'index'
    df_copy = df_copy.reset_index()
    # Rename it to the name specified in our parameters
    df_copy = df_copy.rename(columns={'index': ts_col_name})
    return df_copy

# --- Main Nodes (Updated) ---

# The data splitting and drift introduction nodes remain the same.

def detect_univariate_drift(
    reference_df: pd.DataFrame, 
    analysis_df: pd.DataFrame, 
    params: Dict
) -> pd.DataFrame:
    """Calculates univariate drift using the DataFrame's index as the timestamp."""
    
    ts_col = params['timestamp_column']
    
    # Use the helper to prepare data for NannyML
    reference_df_nml = _add_timestamp_from_index(reference_df, ts_col)
    analysis_df_nml = _add_timestamp_from_index(analysis_df, ts_col)

    print("Detecting univariate drift...")
    calc = nml.UnivariateDriftCalculator(
        column_names=params['features_to_monitor'],
        timestamp_column_name=ts_col,
        chunk_size=params['chunk_size']
    ).fit(reference_data=reference_df_nml)
    
    results = calc.calculate(data=analysis_df_nml)
    return results

def estimate_regression_performance(
    reference_df: pd.DataFrame, 
    analysis_df: pd.DataFrame, 
    params: Dict
) -> pd.DataFrame:
    """Estimates regression performance using the DataFrame's index as the timestamp."""
    
    ts_col = params['timestamp_column']
    
    # Use the helper to prepare data for NannyML
    reference_df_nml = _add_timestamp_from_index(reference_df, ts_col)
    analysis_df_nml = _add_timestamp_from_index(analysis_df, ts_col)
    
    print("Estimating regression performance...")
    perf_calc = nml.RegressionPerformanceCalculator(
        y_true=params['target_column'],
        y_pred=params['prediction_column'],
        timestamp_column_name=ts_col,
        metrics=['rmse', 'mae'],
        chunk_size=params['chunk_size']
    ).fit(reference_df_nml)
    
    results = perf_calc.calculate(data=analysis_df_nml)
    return results.to_df()

def plot_drift_results(results_df: pd.DataFrame, params: Dict):
    """Plots drift results and returns the figure for saving."""
    print("Plotting drift results...")
    
    # Recreate the NannyML Results object, telling it which column to use for the x-axis
    results_obj = nml.results.Result.from_df(
        results_df,
        timestamp_column_name=params['timestamp_column']
    )
    
    figure = results_obj.plot(kind='drift')
    return figure