"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

import great_expectations as gx

from great_expectations.core.expectation_suite import ExpectationSuite
from great_expectations.core import ExpectationConfiguration

from pathlib import Path

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings



logger = logging.getLogger(__name__)


def get_validation_results(checkpoint_result):
    # validation_result is a dictionary containing one key-value pair
    validation_result_key, validation_result_data = next(iter(checkpoint_result["run_results"].items()))

    # Accessing the 'actions_results' from the validation_result_data
    validation_result_ = validation_result_data.get('validation_result', {})

    # Accessing the 'results' from the validation_result_data
    results = validation_result_["results"]
    meta = validation_result_["meta"]
    use_case = meta.get('expectation_suite_name')
    
    
    df_validation = pd.DataFrame({},columns=["Success","Expectation Type","Column","Column Pair","Max Value",\
                                       "Min Value","Element Count","Unexpected Count","Unexpected Percent","Value Set","Unexpected Value","Observed Value"])
    
    
    for result in results:
        # Process each result dictionary as needed
        success = result.get('success', '')
        expectation_type = result.get('expectation_config', {}).get('expectation_type', '')
        column = result.get('expectation_config', {}).get('kwargs', {}).get('column', '')
        column_A = result.get('expectation_config', {}).get('kwargs', {}).get('column_A', '')
        column_B = result.get('expectation_config', {}).get('kwargs', {}).get('column_B', '')
        value_set = result.get('expectation_config', {}).get('kwargs', {}).get('value_set', '')
        max_value = result.get('expectation_config', {}).get('kwargs', {}).get('max_value', '')
        min_value = result.get('expectation_config', {}).get('kwargs', {}).get('min_value', '')

        element_count = result.get('result', {}).get('element_count', '')
        unexpected_count = result.get('result', {}).get('unexpected_count', '')
        unexpected_percent = result.get('result', {}).get('unexpected_percent', '')
        observed_value = result.get('result', {}).get('observed_value', '')
        if type(observed_value) is list:
            #sometimes observed_vaue is not iterable
            unexpected_value = [item for item in observed_value if item not in value_set]
        else:
            unexpected_value=[]
        
        df_validation = pd.concat([df_validation, pd.DataFrame.from_dict( [{"Success" :success,"Expectation Type" :expectation_type,"Column" : column,"Column Pair" : (column_A,column_B),"Max Value" :max_value,\
                                           "Min Value" :min_value,"Element Count" :element_count,"Unexpected Count" :unexpected_count,"Unexpected Percent":unexpected_percent,\
                                                  "Value Set" : value_set,"Unexpected Value" :unexpected_value ,"Observed Value" :observed_value}])], ignore_index=True)
        
    return df_validation


def test_data(df):
    # Add this at the beginning of the function to see what columns we actually have
    logger.info(f"Actual columns in dataset: {df.columns.tolist()}")
    logger.info(f"Column count: {len(df.columns)}")

    context = gx.get_context(context_root_dir="gx")
    datasource_name = "wine_datasource"
    try:
        datasource = context.sources.add_pandas(datasource_name)
        logger.info("Data Source created.")
    except:
        logger.info("Data Source already exists.")
        datasource = context.datasources[datasource_name]

    suite_wine = context.add_or_update_expectation_suite(expectation_suite_name="Wine")

    # SCHEMA VALIDATION - Check data types for all columns
    suite_wine.add_expectation(ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_of_type",
        kwargs={"column": "points", "type_": "int64"},
    ))
    
    suite_wine.add_expectation(ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_of_type",
        kwargs={"column": "price", "type_": "float64"},
    ))
    
    suite_wine.add_expectation(ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_of_type",
        kwargs={"column": "country", "type_": "object"},
    ))
    
    suite_wine.add_expectation(ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_of_type",
        kwargs={"column": "province", "type_": "object"},
    ))
    
    suite_wine.add_expectation(ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_of_type",
        kwargs={"column": "region_1", "type_": "object"},
    ))
    
    suite_wine.add_expectation(ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_of_type",
        kwargs={"column": "variety", "type_": "object"},
    ))
    
    # Check expected column count
    suite_wine.add_expectation(ExpectationConfiguration(
        expectation_type="expect_table_column_count_to_equal",
        kwargs={"value": 8},  
    ))
    
    # VALUE RANGE VALIDATIONS
    suite_wine.add_expectation(ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={
            "column": "points",
            "max_value": 100,
            "min_value": 0
        },
    ))
    
    # Check that price is greater than 0
    suite_wine.add_expectation(ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={
            "column": "price",
            "min_value": 0,
            "strict_min": True  # This ensures price > 0, not just >= 0
        },
    ))
    
    # Check for duplicates - expect the count of unique rows to equal total rows
    suite_wine.add_expectation(ExpectationConfiguration(
        expectation_type="expect_table_row_count_to_equal_other_table",
        kwargs={
            "other_table_row_count": len(df.drop_duplicates())
        },
    ))

    context.add_or_update_expectation_suite(expectation_suite=suite_wine)

    data_asset_name = "test"
    try:
        data_asset = datasource.add_dataframe_asset(name=data_asset_name, dataframe=df)
    except:
        logger.info("The data asset already exists. The required one will be loaded.")
        data_asset = datasource.get_asset(data_asset_name)

    batch_request = data_asset.build_batch_request(dataframe=df)

    checkpoint = gx.checkpoint.SimpleCheckpoint(
        name="checkpoint_wine",  # Changed from marital to wine
        data_context=context,
        validations=[
            {
                "batch_request": batch_request,
                "expectation_suite_name": "Wine",
            },
        ],
    )
    checkpoint_result = checkpoint.run()

    df_validation = get_validation_results(checkpoint_result)
    
    # Direct pandas-based assertions for critical validations
    pd_df_ge = gx.from_pandas(df)

    # Schema validation
    assert pd_df_ge.expect_column_values_to_be_of_type("points", "int64").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("price", "float64").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("country", "object").success == True
    
    # Check for duplicates
    assert len(df) == len(df.drop_duplicates()), "Duplicates found in the dataset"
    
    # Check price is greater than 0
    assert (df["price"] > 0).all(), "Some price values are not greater than 0"
    
    # Check expected column count
    expected_columns = ["points", "price", "country", "province", "region_1", "variety", "taster_name"]
    for column in expected_columns:
        assert column in df.columns, f"Expected column {column} not found in dataset"
    
    assert pd_df_ge.expect_table_column_count_to_equal(8).success == True, "Column count does not match expected value"

    log = logging.getLogger(__name__)
    log.info("Data passed all unit tests successfully")
  
    return df_validation