"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

# Import only the dispatcher function, as it will handle calling the others
from .nodes import dispatch_feature_selection


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=dispatch_feature_selection, # <--- Changed to dispatch_feature_selection
                inputs=[
                    "X_train_data",
                    "y_train_data",
                    "params:feature_selection_params" # <--- Pass the nested parameters
                ],
                outputs="best_columns",
                name="model_feature_selection",
            ),
            # You might also want a node here to filter X_train_data using best_columns
            # For example:
            node(
                func=lambda X, cols: X[cols], # Simple lambda to select columns
                inputs=["X_train_data", "best_columns"],
                outputs="X_train_selected_features", # New dataset for filtered X_train
                name="filter_training_data_with_selected_features",
            ),
        ]
    )
