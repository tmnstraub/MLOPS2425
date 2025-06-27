"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import  split_random


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=split_random,
                inputs="wine_ingested_data", # Update this input name if different in your project
                outputs=["train_data", "batch_data"],
                name="split_data_node",
            ),
        ]
    )
    
