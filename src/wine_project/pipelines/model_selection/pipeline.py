"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import model_selection


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=model_selection,
                inputs=["X_train_engineered","X_test_engineered","y_train","y_test",
                        "production_model_metrics",
                        "production_model",
                        "parameters"],
                outputs="champion_model",
                name="model_selection",
            ),
        ]
    )
