# """Project pipelines."""
# from __future__ import annotations

# from kedro.framework.project import find_pipelines
# from kedro.pipeline import Pipeline


# def register_pipelines() -> dict[str, Pipeline]:
#     """Register the project's pipelines.

#     Returns:
#         A mapping from pipeline names to ``Pipeline`` objects.
#     """
#     pipelines = find_pipelines()
#     pipelines["__default__"] = sum(pipelines.values())
#     return pipelines

"""Project pipelines."""
from typing import Dict
from kedro.pipeline import Pipeline, pipeline

from wine_project.pipelines import (
    ingestion as data_ingestion,
    data_unit_tests as data_tests,
    data_preprocessing,
    split_train_pipeline as split_train,
    # model_selection,
    # model_train,
    # feature_selection,
    model_predict,
    feature_engineering,
    data_drift
)

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    ingestion_pipeline = data_ingestion.create_pipeline()
    feature_engineering_pipeline = feature_engineering.create_pipeline()
    data_unit_tests_pipeline = data_tests.create_pipeline()
    split_train_pipeline = split_train.create_pipeline()
    # model_train_pipeline = model_train.create_pipeline()
    # model_selection_pipeline = model_selection.create_pipeline()
    # feature_selection_pipeline = feature_selection.create_pipeline()
    model_predict_pipeline = model_predict.create_pipeline()
    data_preprocessing_pipeline = data_preprocessing.create_pipeline()
    data_drift_pipeline = data_drift.create_pipeline()

    # Create a combined pipeline with all nodes that shows the sequential workflow
    # This allows visualizing the entire ML workflow in Kedro-Viz
    combined_pipeline = ingestion_pipeline + data_unit_tests_pipeline + data_preprocessing_pipeline + split_train_pipeline + feature_engineering_pipeline + model_predict_pipeline + data_drift_pipeline
    
    return {
        "ingestion": ingestion_pipeline,
        "data_unit_tests": data_unit_tests_pipeline,
        "data_preprocessing": data_preprocessing_pipeline,
        "split_train": split_train_pipeline,
        "feature_engineering": feature_engineering_pipeline,
        'data_drift': data_drift_pipeline,
        # "feature_selection": feature_selection_pipeline,
        # "model_selection": model_selection_pipeline,
        # "model_train": model_train_pipeline,
        "inference": model_predict_pipeline,
        # Add default pipeline that combines all implemented pipelines in sequence
        "__default__": combined_pipeline
    }