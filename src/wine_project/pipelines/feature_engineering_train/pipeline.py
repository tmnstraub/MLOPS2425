"""
Pipeline for engineering features for the training dataset
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import engineer_train_features, create_one_hot_encoded_features


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=engineer_train_features,
                inputs=["train_preprocessed"],
                outputs="train_feature_engineered",
                name="engineer_train_features_node",
            ),
            node(
                func=create_one_hot_encoded_features,
                inputs=["train_feature_engineered"],
                outputs="train_feature_engineered_one_hot",
                name="one_hot_encode_train_features_node",
            ),
        ]
    )
