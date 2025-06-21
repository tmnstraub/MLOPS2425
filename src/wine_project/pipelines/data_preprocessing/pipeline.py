"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.19.13
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import na_col_to_unknown


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
                node(
                func= na_col_to_unknown,
                inputs={
                    "parameters": "params:columns_to_fill",
                    "df": "wine_ingested_data"
                },
                outputs= "wine_preprocessed",
                name="drop_col",
            ),
    ])
