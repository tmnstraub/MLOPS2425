"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.19.13
"""

import pandas as pd 
from great_expectations.core import ExpectationSuite, ExpectationConfiguration
import typing as t

def drop_col(parameters: t.Dict, df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns = parameters)

    