"""
Pipeline for preprocessing batch data
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_batch_data


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=preprocess_batch_data,
                inputs="batch_data",
                outputs="batch_preprocessed",
                name="preprocess_batch_data_node",
            ),
        ]
    )

