from kedro.pipeline import Pipeline, node
from .nodes import engineer_features

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=engineer_features,
            inputs="X_train",  # Changed from model_input_table to X_train
            outputs="X_train_engineered",
            name="feature_engineering_train_node"
        ),
        node(
            func=engineer_features,
            inputs="X_test",  # Apply same transformations to test data
            outputs="X_test_engineered",
            name="feature_engineering_test_node"
        )
    ])