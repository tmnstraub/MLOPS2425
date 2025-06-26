"""
This is a boilerplate pipeline 'data_drift'
generated using Kedro 0.19.13
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    # split_reference_analysis, 
    # introduce_drift, 
    detect_univariate_drift, 
    plot_drift_results,
    # Make sure to import all the nodes you use
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        # node(
        #     func=split_reference_analysis,
        #     inputs="original_data",
        #     outputs=["reference_data", "analysis_data_raw"],
        #     name="split_data_node"
        # ),
        # node(
        #     func=introduce_drift,
        #     inputs="analysis_data_raw",
        #     outputs="analysis_data",
        #     name="introduce_drift_node"
        # ),
        node(
            func=detect_univariate_drift,
            inputs=["wine_raw_data", "wine_data_drift_test", "params:data_drift"],
            outputs="univariate_drift_results",
            name="detect_drift_node"
        ),
        node(
            func=plot_drift_results,
            # Add the parameters as an input to the plotting node
            inputs=["univariate_drift_results", "params:data_drift"],
            outputs="univariate_drift_plot",
            name="plot_drift_node"
        ),
        # You would do the same for your performance estimation and plotting nodes
    ])