"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.19.13
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import drop_col


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
                node(
                func= drop_col,
                inputs={
                    "parameters": "params:drop_col_options",
                    "df": "wine_raw_data"
                },
                outputs= "wine_preprocessed",
                name="drop_col",
            ),
    ])
