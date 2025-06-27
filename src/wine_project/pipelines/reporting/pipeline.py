"""
This is a boilerplate pipeline 'reporting'
generated using Kedro 0.19.5
"""

"""
Pipeline for generating reports and visualizations
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import visualize_data_unit_test_results

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=visualize_data_unit_test_results,
                inputs="reporting_data_unit_test",
                outputs="reporting_data_unit_test_visualized",
                name="visualize_data_unit_test_results_node",
            ),
        ]
    )