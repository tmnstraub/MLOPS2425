"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def split_random(df):

    ref_data = df.sample(frac=0.8,random_state=200)
    ana_data = df.drop(ref_data.index)

    return ref_data, ana_data