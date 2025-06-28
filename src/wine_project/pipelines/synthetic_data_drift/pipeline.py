"""
This is a boilerplate pipeline 'synthetic_data_drift'
generated using Kedro 0.19.14
"""

from kedro.pipeline import node, Pipeline, pipeline  
from .nodes import apply_points_drift, save_drifted_data, load_and_sample_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        # Step 1: Load and sample the raw wine data
        node(
            func=load_and_sample_data,
            inputs=["wine_raw_data", "params:n_samples", "params:random_state"],
            outputs="sampled_wine_data",
            name="load_and_sample_data_node",
        ),
        # Step 2: Apply points drift to the sampled data
        node(
            func=apply_points_drift,
            inputs=[
                "sampled_wine_data",
                "params:drift_loc",
                "params:drift_scale",
                "params:points_column",
                "params:clip_values"
            ],
            outputs="drifted_data",
            name="apply_points_drift_node",
        ),
        # Step 3: Save the drifted data
        node(
            func=save_drifted_data,
            inputs="drifted_data",
            outputs="wine_data_drift_test",
            name="save_drifted_data_node",
        ),
    ])
