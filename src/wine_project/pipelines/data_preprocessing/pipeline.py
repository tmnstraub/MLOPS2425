"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.19.13
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import na_col_to_unknown, drop_row


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([

                node(
                func = drop_row,
                inputs={
                    "parameters": "params:na_values_col_drop",
                    "df": "wine_ingested_data"
                },
                outputs="wine_preprocessed_intermediate",
                name="drop_row_na_values"),

                node(
                func= na_col_to_unknown,
                inputs={
                    "parameters": "params:columns_to_fill",
                    "df": "wine_preprocessed_intermediate"
                },
                outputs= "wine_preprocessed",
                name="na_col_values_to_unknown",
            ),

                
            
    ])
