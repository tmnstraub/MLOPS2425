from kedro.pipeline import Pipeline, node
from .nodes import engineer_features

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=engineer_features,
            inputs="model_input_table",  # or adjust this to your dataset name
            outputs="feature_table",
            name="feature_engineering_node"
        )
    ])
