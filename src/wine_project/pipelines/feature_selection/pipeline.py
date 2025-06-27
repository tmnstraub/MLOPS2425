
"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import feature_selection


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=feature_selection,
                inputs=["X_train_one_hot","y_train_one_hot","parameters"],
                outputs="best_columns",
                name="model_feature_selection",
            ),
        ]
    )
