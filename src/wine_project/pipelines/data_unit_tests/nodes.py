"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

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
    context = gx.get_context(context_root_dir = "//..//..//gx")
    datasource_name = "bank_datasource"
    try:
        datasource = context.sources.add_pandas(datasource_name)
        logger.info("Data Source created.")
    except:
        logger.info("Data Source already exists.")
        datasource = context.datasources[datasource_name]

    suite_bank = context.add_or_update_expectation_suite(expectation_suite_name="Bank")
    
    #add more expectations to your data
    expectation_marital = ExpectationConfiguration(
    expectation_type="expect_column_distinct_values_to_be_in_set",
    kwargs={
        "column": "marital",
        "value_set" : ['married', 'single', 'divorced']
    },
        )
    suite_bank.add_expectation(expectation_configuration=expectation_marital)

    expectation_balance = ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={
            "column": "balance",
            "max_value": 105000,
            "min_value": 0
        },
    )
    suite_bank.add_expectation(expectation_configuration=expectation_balance)

    expectation_age = ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={
            "column": "age",
            "max_value": 100,
            "min_value": 18
        },
    )
    suite_bank.add_expectation(expectation_configuration=expectation_age)


    context.add_or_update_expectation_suite(expectation_suite=suite_bank)

    data_asset_name = "test"
    try:
        data_asset = datasource.add_dataframe_asset(name=data_asset_name, dataframe= df)
    except:
        logger.info("The data asset alread exists. The required one will be loaded.")
        data_asset = datasource.get_asset(data_asset_name)

    batch_request = data_asset.build_batch_request(dataframe= df)


    checkpoint = gx.checkpoint.SimpleCheckpoint(
        name="checkpoint_marital",
        data_context=context,
        validations=[
            {
                "batch_request": batch_request,
                "expectation_suite_name": "Bank",
            },
        ],
    )
    checkpoint_result = checkpoint.run()

    df_validation = get_validation_results(checkpoint_result)
    #base on these results you can make an assert to stop your pipeline

    pd_df_ge = gx.from_pandas(df)

    assert pd_df_ge.expect_column_values_to_be_of_type("duration", "int64").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("marital", "str").success == True
    #assert pd_df_ge.expect_table_column_count_to_equal(23).success == False

    log = logging.getLogger(__name__)
    log.info("Data passed on the unit data tests")
  

    return df_validation