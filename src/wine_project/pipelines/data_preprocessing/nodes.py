"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.19.13
"""

import pandas as pd 
from great_expectations.core import ExpectationSuite, ExpectationConfiguration
import typing as t

# def drop_col(parameters: t.Dict, df: pd.DataFrame) -> pd.DataFrame:
    # return df.drop(columns = parameters)


def na_col_to_unknown(df: pd.DataFrame, parameters: t.Dict[str, t.Any]) -> pd.DataFrame:
   
    for col in parameters:
        # It's good practice to check if the column exists in the DataFrame
        if col in df.columns:
            df[col] = df[col].fillna('unknown')
        else:
            # You could add a log or a print statement here to warn about missing columns
            print(f"Warning: Column '{col}' not found in the DataFrame and was skipped.")

    return df
    