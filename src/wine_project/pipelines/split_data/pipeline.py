
"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import  split_random


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func= split_random,
                inputs= "ingested_data",
                outputs=["ref_data","ana_data"],
                name="split_out_of_sample",
            ),
        ]
    )
