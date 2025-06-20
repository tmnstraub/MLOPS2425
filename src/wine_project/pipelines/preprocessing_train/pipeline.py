
"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import feature_engineer, clean_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [


            node(
                func= clean_data,
                inputs="ref_data",
                outputs= ["ref_data_cleaned","reporting_data_train"],
                name="clean_data",
            ),
            node(
                func= feature_engineer,
                inputs="ref_data_cleaned",
                outputs= ["preprocessed_training_data","encoder_transform"],
                name="preprocessed_training",
            ),

        ]
    )
