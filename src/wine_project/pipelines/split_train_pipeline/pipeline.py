
"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import  split_data_and_validate


# src/<your_project_name>/pipelines/data_science/pipeline.py

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import split_data_and_validate # Import the updated function

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # ... an upstream node that provides the 'processed_data' DataFrame
            node(
                func=split_data_and_validate,
                inputs=["wine_preprocessed", "params:split_data_and_validate"],
                outputs=[
                    "X_train",
                    "X_test",
                    "y_train",
                    "y_test",
                    "feature_names"
                ],
                name="split_data_and_validate_node",
            ),
            # ... downstream nodes (e.g., model training) that consume the validated
            # splits like X_train, y_train, etc.
        ]
    )