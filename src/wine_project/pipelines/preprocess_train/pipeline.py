"""
Pipeline for preprocessing training data
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_train_data


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=preprocess_train_data,
                inputs="train_data",
                outputs="train_preprocessed",
                name="preprocess_train_data_node",
            ),
        ]
    )
