"""
Pipeline for engineering features for the training dataset
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    engineer_train_features, 
    select_train_features,
    create_one_hot_encoded_features
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the feature engineering pipeline for training data."""
    return pipeline(
        [
            # First node: Create all engineered features and store in train_all_features dataset
            node(
                func=engineer_train_features,
                inputs="train_preprocessed",
                outputs="train_all_features",
                name="engineer_train_all_features_node",
            ),
            # Second node: Select features based on correlation
            node(
                func=select_train_features,
                inputs=["train_all_features", "params:features_to_drop"],
                outputs="train_feature_engineered",
                name="select_train_features_node",
            ),
            # Third node: Create one-hot encoded features from selected features
            node(
                func=create_one_hot_encoded_features,
                inputs="train_feature_engineered",
                outputs="train_feature_engineered_one_hot",
                name="one_hot_encode_train_features_node",
            ),
        ]
    )
