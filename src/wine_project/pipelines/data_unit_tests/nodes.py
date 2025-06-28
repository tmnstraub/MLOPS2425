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

# Define a dictionary mapping expectation types to human-readable descriptions
EXPECTATION_DESCRIPTIONS = {
    "expect_table_column_count_to_equal": "Number of features in dataset",
    "expect_column_values_to_be_between": "Value range check",
    "expect_column_values_to_be_of_type": "Data type check",
    "expect_table_row_count_to_equal_other_table": "No duplicate records check"
}

logger = logging.getLogger(__name__)

def check_feature_count(suite, expected_count):
    """Verify that the dataset has the expected number of features."""
    logger.info(f"Testing: Number of features should be {expected_count}")
    suite.add_expectation(ExpectationConfiguration(
        expectation_type="expect_table_column_count_to_equal",
        kwargs={"value": expected_count},
    ))
    return suite

def check_numeric_range(suite, column_name, min_value, max_value=None, strict_min=False):
    """Verify that numeric values in a column fall within the expected range."""
    range_desc = f">{min_value}" if strict_min else f">={min_value}"
    if max_value is not None:
        range_desc += f" and <={max_value}"
        
    logger.info(f"Testing: '{column_name}' values should be {range_desc}")
    
    kwargs = {"column": column_name}
    if min_value is not None:
        kwargs["min_value"] = min_value
        kwargs["strict_min"] = strict_min
    if max_value is not None:
        kwargs["max_value"] = max_value
        
    suite.add_expectation(ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs=kwargs,
    ))
    return suite

def check_column_type(suite, column_name, expected_type):
    """Verify that a column has the expected data type."""
    logger.info(f"Testing: '{column_name}' should be of type {expected_type}")
    suite.add_expectation(ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_of_type",
        kwargs={"column": column_name, "type_": expected_type},
    ))
    return suite

def check_no_duplicates(suite, df):
    """Verify that there are no duplicate rows in the dataset."""
    logger.info("Testing: Dataset should not contain duplicate records")
    suite.add_expectation(ExpectationConfiguration(
        expectation_type="expect_table_row_count_to_equal_other_table",
        kwargs={
            "other_table_row_count": len(df.drop_duplicates())
        },
    ))
    return suite

def enhance_validation_results(df_validation):
    """Add human-readable descriptions to validation results."""
    # Add a more readable description column
    df_validation['Test Description'] = df_validation['Expectation Type'].map(EXPECTATION_DESCRIPTIONS)
    
    # Add more context based on column and parameters
    for idx, row in df_validation.iterrows():
        if not pd.isna(row['Column']) and row['Column']:
            df_validation.at[idx, 'Test Description'] += f" ({row['Column']})"
        
        if row['Expectation Type'] == "expect_column_values_to_be_between":
            min_val = row['Min Value'] if not pd.isna(row['Min Value']) else "any"
            max_val = row['Max Value'] if not pd.isna(row['Max Value']) else "any"
            df_validation.at[idx, 'Test Description'] += f" [{min_val} to {max_val}]"
    
    return df_validation

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
    # Log actual dataset properties for debugging
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

    # Create the expectation suite
    suite_wine = context.add_or_update_expectation_suite(expectation_suite_name="Wine")

    # Use helper functions to add expectations with readable descriptions
    suite_wine = check_feature_count(suite_wine, 8)  # Expect 8 columns
    
    # Check data types
    suite_wine = check_column_type(suite_wine, "points", "int64")
    suite_wine = check_column_type(suite_wine, "price", "float64")
    suite_wine = check_column_type(suite_wine, "country", "object")
    suite_wine = check_column_type(suite_wine, "province", "object")
    suite_wine = check_column_type(suite_wine, "region_1", "object")
    suite_wine = check_column_type(suite_wine, "variety", "object")
    
    # Check value ranges
    suite_wine = check_numeric_range(suite_wine, "points", 0, 100)
    suite_wine = check_numeric_range(suite_wine, "price", 0, strict_min=True)  # Price > 0
    
    # Check for duplicates
    suite_wine = check_no_duplicates(suite_wine, df)

    # Save the expectation suite
    context.add_or_update_expectation_suite(expectation_suite=suite_wine)

    # Create data asset and run validations
    data_asset_name = "test"
    try:
        data_asset = datasource.add_dataframe_asset(name=data_asset_name, dataframe=df)
    except:
        logger.info("The data asset already exists. The required one will be loaded.")
        data_asset = datasource.get_asset(data_asset_name)

    batch_request = data_asset.build_batch_request(dataframe=df)

    checkpoint = gx.checkpoint.SimpleCheckpoint(
        name="checkpoint_wine",
        data_context=context,
        validations=[
            {
                "batch_request": batch_request,
                "expectation_suite_name": "Wine",
            },
        ],
    )
    checkpoint_result = checkpoint.run()

    # Get validation results and enhance them with readable descriptions
    df_validation = get_validation_results(checkpoint_result)
    df_validation = enhance_validation_results(df_validation)
    
    # Direct pandas-based assertions for critical validations
    pd_df_ge = gx.from_pandas(df)

    # Check expected column count
    expected_columns = ["points", "price", "country", "province", "region_1", "variety", "taster_name", "datetime"]
    for column in expected_columns:
        assert column in df.columns, f"Expected column {column} not found in dataset"
    
    assert pd_df_ge.expect_table_column_count_to_equal(8).success == True, "Column count does not match expected value"
    
    # Check for duplicates
    assert len(df) == len(df.drop_duplicates()), "Duplicates found in the dataset"
    
    # Check price is greater than 0
    assert (df["price"] > 0).all(), "Some price values are not greater than 0"

    logger.info("Data passed all unit tests successfully")
  
    return df_validation