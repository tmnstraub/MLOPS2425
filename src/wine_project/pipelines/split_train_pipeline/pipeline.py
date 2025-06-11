
"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import  split_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func= split_data,
                inputs=["preprocessed_training_data","parameters"],
                outputs= ["X_train_data","X_test_data","y_train_data","y_test_data","best_columns"],
                name="split",
            ),
        ]
    )
