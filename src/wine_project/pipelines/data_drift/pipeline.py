from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    detect_univariate_drift, 
    generate_drift_plot_image,
    # include other nodes as needed
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=detect_univariate_drift,
            inputs=["wine_raw_data", "wine_data_drift_test", "params:data_drift"],
            outputs=["univariate_drift_df", "univariate_drift_results"],
            name="detect_univariate_drift_node"
        ),
        node(
            func=generate_drift_plot_image,
            inputs="univariate_drift_results",
            outputs="univariate_drift_plot",
            name="generate_univariate_drift_plot_node"
        ),
    ])