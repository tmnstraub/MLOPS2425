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
    preprocessing_train as preprocess_train,
    split_train_pipeline as split_train,
    model_selection as model_selection_pipeline,
    model_train as model_train_pipeline,
    feature_selection as feature_selection_pipeline,
    split_data,
    preprocessing_batch,
    model_predict

)

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    ingestion_pipeline = data_ingestion.create_pipeline()
    data_unit_tests_pipeline = data_tests.create_pipeline()
    split_data_pipeline = split_data.create_pipeline()
    preprocess_train_pipeline = preprocess_train.create_pipeline()
    split_train_pipeline = split_train.create_pipeline()
    model_train = model_train_pipeline.create_pipeline()
    model_selection = model_selection_pipeline.create_pipeline()
    feature_selection = feature_selection_pipeline.create_pipeline()
    preprocess_batch_pipeline = preprocessing_batch.create_pipeline()
    model_predict_pipeline = model_predict.create_pipeline()

    return {
        "ingestion": ingestion_pipeline,
        "data_unit_tests": data_unit_tests_pipeline,
        "split_data": split_data_pipeline,
        "preprocess_train": preprocess_train_pipeline,
        "split_train": split_train_pipeline,
        "model_selection": model_selection,
        "model_train": model_train,
        "feature_selection":feature_selection,
        "production_full_train_process" : preprocess_train_pipeline + split_train_pipeline + model_train,
        "preprocess_batch": preprocess_batch_pipeline,
        "inference" : model_predict_pipeline,
        "production_full_prediction_process" : preprocess_batch_pipeline + model_predict_pipeline,
  
    }