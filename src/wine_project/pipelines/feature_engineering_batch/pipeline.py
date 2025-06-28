"""
Pipeline for engineering features for the batch dataset
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    engineer_batch_features,
    select_batch_features,
    create_one_hot_encoded_features
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the feature engineering pipeline for batch data."""
    return pipeline(
        [
            # First node: Create all engineered features
            node(
                func=engineer_batch_features,
                inputs="batch_preprocessed",
                outputs="batch_all_features",
                name="engineer_batch_all_features_node",
            ),
            # Second node: Select features (drop the same features as in train pipeline)
            node(
                func=select_batch_features,
                # Pass both the dataset and the parameters
                inputs=["batch_all_features", "params:features_to_drop"],
                outputs="batch_feature_engineered",
                name="select_batch_features_node",
            ),
            # Third node: Create one-hot encoded features from selected features
            node(
                func=create_one_hot_encoded_features,
                inputs="batch_feature_engineered",
                outputs="batch_feature_engineered_one_hot",
                name="one_hot_encode_batch_features_node",
            ),
        ]
    )
