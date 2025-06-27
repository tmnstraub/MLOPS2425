"""
Pipeline for engineering features for the batch dataset
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import engineer_batch_features, create_one_hot_encoded_features


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=engineer_batch_features,
                inputs=["batch_preprocessed"],
                outputs="batch_feature_engineered",
                name="engineer_batch_features_node",
            ),
            node(
                func=create_one_hot_encoded_features,
                inputs=["batch_feature_engineered"],
                outputs="batch_feature_engineered_one_hot",
                name="one_hot_encode_batch_features_node",
            ),
        ]
    )
