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
    reporting,
    train_batch_split, 
    # data_preprocessing,  # Remove this import since it doesn't exist
    train_val_split,
    preprocess_train,
    preprocess_batch,
    feature_engineering_train,
    feature_engineering_batch,
    model_selection,
    # model_train,
    feature_selection,
    model_predict
)

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    ingestion_pipeline = data_ingestion.create_pipeline()
    feature_engineering_train_pipeline = feature_engineering_train.create_pipeline()
    feature_engineering_batch_pipeline = feature_engineering_batch.create_pipeline()
    data_unit_tests_pipeline = data_tests.create_pipeline()
    reporting_pipeline = reporting.create_pipeline()
    train_batch_split_pipeline = train_batch_split.create_pipeline()
    preprocess_train_pipeline = preprocess_train.create_pipeline()
    preprocess_batch_pipeline = preprocess_batch.create_pipeline()
    train_val_split_pipeline = train_val_split.create_pipeline()
    model_train_pipeline = model_train.create_pipeline()
    model_selection_pipeline = model_selection.create_pipeline()
    feature_selection_pipeline = feature_selection.create_pipeline()
    model_predict_pipeline = model_predict.create_pipeline()
    # data_preprocessing_pipeline = data_preprocessing.create_pipeline()  # Remove this line

    # Create a combined pipeline with all nodes that shows the sequential workflow
    # This allows visualizing the entire ML workflow in Kedro-Viz
    combined_pipeline = (
        ingestion_pipeline + 
        data_unit_tests_pipeline + 
        train_batch_split_pipeline + 
        preprocess_train_pipeline + 
        preprocess_batch_pipeline + 
        feature_engineering_train_pipeline +
        feature_engineering_batch_pipeline +
        train_val_split_pipeline + 
        model_predict_pipeline
    )
    
    # Create smaller logical pipeline groups for easier execution
    data_preparation_pipeline = ingestion_pipeline + data_unit_tests_pipeline
    
    train_batch_pipeline = train_batch_split_pipeline
    
    preprocessing_pipeline = preprocess_train_pipeline + preprocess_batch_pipeline
    
    feature_eng_pipeline = feature_engineering_train_pipeline + feature_engineering_batch_pipeline
    
    return {
        "ingestion": ingestion_pipeline,
        "data_unit_tests": data_unit_tests_pipeline,
        "reporting": reporting_pipeline,
        "data_quality": data_tests.create_pipeline() + reporting_pipeline,
        "train_batch_split": train_batch_split_pipeline,
        "preprocess_train": preprocess_train_pipeline,
        "preprocess_batch": preprocess_batch_pipeline,
        "feature_engineering_train": feature_engineering_train_pipeline,
        "feature_engineering_batch": feature_engineering_batch_pipeline,
        "train_val_split": train_val_split_pipeline,
        "model_selection": model_selection_pipeline,
        "feature_selection": feature_selection_pipeline,
        # "model_train": model_train_pipeline,
        "inference": model_predict_pipeline,
        # Add default pipeline that combines all implemented pipelines in sequence
        "__default__": combined_pipeline,
        # Add logical pipeline groups
        "data_preparation": data_preparation_pipeline,
        "train_batch": train_batch_pipeline,
        "preprocessing": preprocessing_pipeline,
        "feature_eng": feature_eng_pipeline,
    }