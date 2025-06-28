# In src/<your_package>/pipelines/data_drift/nodes.py

import pandas as pd
import nannyml as nml
from typing import Dict, Tuple
import io
from PIL import Image

# --- Helper Function (Unchanged) ---
def _add_timestamp_from_index(df: pd.DataFrame, ts_col_name: str) -> pd.DataFrame:
    """
    Takes a DataFrame, creates a copy, and adds a new column 
    named `ts_col_name` from the DataFrame's index.
    """
    df_copy = df.copy()
    df_copy = df_copy.reset_index()
    df_copy = df_copy.rename(columns={'index': ts_col_name})
    return df_copy

# --- Main Nodes (Updated) ---

def detect_univariate_drift(
    reference_df: pd.DataFrame, 
    analysis_df: pd.DataFrame, 
    params: Dict
):  # CHANGED: Output signature
    """
    Calculates univariate drift and returns both the results as a 
    DataFrame and the complete NannyML Result object for plotting.
    """
    ts_col = params['timestamp_column']
    reference_df_nml = _add_timestamp_from_index(reference_df, ts_col)
    analysis_df_nml = _add_timestamp_from_index(analysis_df, ts_col)

    print("Detecting univariate drift...")
    calc = nml.UnivariateDriftCalculator(
        column_names=params['features_to_monitor'],
        timestamp_column_name=ts_col,
        continuous_methods=['kolmogorov_smirnov', 'jensen_shannon'],
        chunk_size=params['chunk_size']
    ).fit(reference_data=reference_df_nml)
    
    results = calc.calculate(data=analysis_df_nml)
    
    # CHANGED: Return both the DataFrame and the full result object
    return results.to_df(), results
    
    # CHANGED: Return both the DataFrame and the full result object
    return results.to_df(), results

# NEW NODE: Generates and returns a plot as a savable image object
def generate_drift_plot_image(univariate_drift_results) -> Image.Image:
    """
    Takes NannyML result object, generates a drift plot, and returns it
    as a PIL Image object that can be saved to the Data Catalog.
    """
    print("Generating drift plot image...")
    
    # 1. Generate the Plotly figure
    figure = univariate_drift_results.plot(kind='drift')
    
    # 2. Convert the figure to PNG bytes in memory
    image_bytes = figure.to_image(format='png', width=1200, height=600, scale=2)
    
    # 3. Create a PIL Image object from the bytes
    return Image.open(io.BytesIO(image_bytes))