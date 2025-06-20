"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""
import logging
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder , LabelEncoder


from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

conf_path = str(Path('') / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)
credentials = conf_loader["credentials"]


logger = logging.getLogger(__name__)


"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

import logging
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder , LabelEncoder


def age_(data):
    
    data['bin_age'] = 0  
    data.loc[(data['age'] <= 35) & (data['age'] >= 18),'bin_age'] = 1
    data.loc[(data['age'] <= 60) & (data['age'] >= 36),'bin_age'] = 2
    data.loc[data['age'] >=61,'bin_age'] = 3
    
    return data

def campaign_(data):
    
    
    data.loc[data['campaign'] == 1,'campaign'] = 1
    data.loc[(data['campaign'] >= 2) & (data['campaign'] <= 3),'campaign'] = 2
    data.loc[data['campaign'] >= 4,'campaign'] = 3
    
    return data

def duration_(data):
    
    data['t_min'] = 0
    data['t_e_min'] = 0
    data['e_min']=0
    data.loc[data['duration'] <= 5,'t_min'] = 1
    data.loc[(data['duration'] > 5) & (data['duration'] <= 10),'t_e_min'] = 1
    data.loc[data['duration'] > 10,'e_min'] = 1
    
    return data

def pdays_(data):
    data['pdays_not_contacted'] = 0
    data['months_passed'] = 0
    data.loc[data['pdays'] == -1 ,'pdays_not_contacted'] = 1
    data['months_passed'] = data['pdays']/30
    data.loc[(data['months_passed'] >= 0) & (data['months_passed'] <=2) ,'months_passed'] = 1
    data.loc[(data['months_passed'] > 2) & (data['months_passed'] <=6),'months_passed'] = 2
    data.loc[data['months_passed'] > 6 ,'months_passed'] = 3
    
    return data


def balance_(data):
    data['Neg_Balance'] = 0
    data['No_Balance'] = 0
    data['Pos_Balance'] = 0
    data.loc[~data['balance']<0,'Neg_Balance'] = 1
    data.loc[data['balance'] < 1,'bin_Balance'] = 0
    data.loc[(data['balance'] >= 1) & (data['balance'] < 100),'bin_Balance'] = 1
    data.loc[(data['balance'] >= 100) & (data['balance'] < 500),'bin_Balance'] = 2
    data.loc[(data['balance'] >= 500) & (data['balance'] < 2000),'bin_Balance'] = 3
    data.loc[(data['balance'] >= 2000) & (data['balance'] < 5000),'bin_Balance'] = 4
    data.loc[data['balance'] >= 5000,'bin_Balance'] = 5
    
    return data


def feature_engineer( data: pd.DataFrame, OH_encoder) -> pd.DataFrame:
    
    log = logging.getLogger(__name__)

    df = data.copy()
    df.fillna(-9999,inplace=True)
    le = LabelEncoder()

    df = campaign_(df)
    df = age_(df)
    df = balance_(df)
    if "y" in df.columns:
        df["y"] = df["y"].map({"no":0, "yes":1})
    
    #new profiling feature
    # In this step we should start to think on feature store
    df["mean_balance_bin_age"] = df.groupby("bin_age")["balance"].transform("mean")
    df["std_balance_bin_age"] = df.groupby("bin_age")["balance"].transform("std")
    df["z_score_bin_age"] = (df["mean_balance_bin_age"] - df["balance"])/(df["std_balance_bin_age"])
    #df['day_of_week'] = le.fit_transform(df['day_of_week'])
    df['month'] = le.fit_transform(df['month'])
    
    
    
    numerical_features = df.select_dtypes(exclude=['object','string','category']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object','string','category']).columns.tolist()

    #Exercise create an assert for numerical and categorical features
    

    OH_cols= pd.DataFrame(OH_encoder.transform(df[categorical_features]))

    # Adding column names to the encoded data set.
    OH_cols.columns = OH_encoder.get_feature_names_out(categorical_features)

    # One-hot encoding removed index; put it back
    OH_cols.index = df.index

    # Remove categorical columns (will replace with one-hot encoding)
    num_df = df.drop(categorical_features, axis=1)

    # Add one-hot encoded columns to numerical features
    df_final = pd.concat([num_df, OH_cols], axis=1)

    log.info(f"The final dataframe has {len(df_final.columns)} columns.")

    return df_final


