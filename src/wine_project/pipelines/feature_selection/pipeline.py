# src/wine_project/pipelines/feature_selection/pipeline.py

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    feature_selection_rfe,
    correlation_feature_selection,
    intersect_features,
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=feature_selection_rfe,
                inputs=[
                    "X_train_data",
                    "y_train_data",
                    "champion_model", # <--- Input from catalog.yml
                    "params:feature_selection_params"
                ],
                outputs="rfe_selected_features",
                name="rfe_feature_selection_node",
            ),
            node(
                func=correlation_feature_selection,
                inputs=[
                    "X_train_data",
                    "y_train_data",
                    "params:feature_selection_params"
                ],
                outputs="corr_selected_features",
                name="correlation_feature_selection_node",
            ),
            node(
                func=intersect_features,
                inputs=["rfe_selected_features", "corr_selected_features"],
                outputs="best_columns", # This is the final output
                name="intersect_feature_lists",
            ),
            node(
                func=lambda df, cols: df[cols],
                inputs=["X_train_data", "best_columns"],
                outputs="X_train_selected", # A more descriptive name
                name="filter_training_data_by_features",
            ),
        ]
    )