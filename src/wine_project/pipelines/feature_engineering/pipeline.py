from kedro.pipeline import Pipeline, node
from .nodes import run_feature_engineering

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=run_feature_engineering,
            inputs="model_input_table",  # or adjust this to your dataset name
            outputs="feature_table",
            name="feature_engineering_node"
        )
    ])
