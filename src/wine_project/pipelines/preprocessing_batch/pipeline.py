
"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import feature_engineer

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func= feature_engineer,
                inputs=["ana_data","encoder_transform"],
                outputs= "preprocessed_batch_data",
                name="preprocessed_batch",
            ),

        ]
    )
