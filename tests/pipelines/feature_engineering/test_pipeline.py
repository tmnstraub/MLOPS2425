import pytest
from src.wine_project.pipelines.feature_engineering.pipeline import create_pipeline

def test_create_pipeline():
    """Test that the feature engineering pipeline is created correctly."""
    # Create the pipeline
    pipeline = create_pipeline()
    
    # Check that the pipeline has the correct number of nodes
    assert len(pipeline.nodes) == 2
    
    # Check the names of the nodes
    node_names = [node.name for node in pipeline.nodes]
    assert "feature_engineering_train_node" in node_names
    assert "feature_engineering_test_node" in node_names
    
    # Check inputs and outputs
    for node in pipeline.nodes:
        if node.name == "feature_engineering_train_node":
            assert node.inputs == ["X_train"]
            assert node.outputs == ["X_train_engineered"]
        elif node.name == "feature_engineering_test_node":
            assert node.inputs == ["X_test"]
            assert node.outputs == ["X_test_engineered"]
