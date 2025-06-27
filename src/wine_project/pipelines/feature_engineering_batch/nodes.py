import pandas as pd
import pycountry_convert as pc
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def remove_index_column(data):
    """
    Remove index column from the DataFrame.
    
    Args:
        data: Pandas DataFrame that might have an index or df_index column
        
    Returns:
        DataFrame with index column removed
    """
    # Check for 'df_index' column
    if 'df_index' in data.columns:
        data = data.drop(columns=['df_index'])
    
    # Check for 'index' column
    if 'index' in data.columns:
        data = data.drop(columns=['index'])
    
    return data

def classify_points(score):
    if score < 85:
        return 'Low'
    elif score < 90:
        return 'Medium'
    else:
        return 'High'

# Wine categorization keywords
red_terms = [
    'red', 'pinot noir', 'pinot nero', 'carmenère', "nero d'avola", 'mourvèdre',
    'blaufränkisch', 'primitivo', 'zinfandel', 'merlot', 'syrah', 'malbec',
    'sangiovese', 'nebbiolo', 'tempranillo', 'touriga nacional', 'tannat',
    'dolcetto', 'pinotage'
]

white_terms = [
    'white', 'pinot bianco', 'pinot blanc', 'grüner veltliner', 'sauvignon blanc',
    'riesling', 'pinot gris', 'pinot grigio', 'melon', 'vermentino', 'sémillon',
    'fiano', 'alvarinho', 'friulano', 'nerello', 'greco', 'grillo'
]

fortified_terms = ['port', 'sherry', 'tokay', 'muscat', 'tawny', 'botrytis']

def classify_wine_type_main(variety):
    # Handle NaN, None, or non-string values
    if pd.isna(variety) or not isinstance(variety, str):
        return 'unknown'
        
    v = variety.lower()
    if any(x in v for x in ['champagne', 'sparkling', 'prosecco']):
        return 'sparkling'
    elif 'rosé' in v or 'rose' in v:
        return 'rosé'
    elif any(x in v for x in fortified_terms):
        return 'fortified'
    elif any(x in v for x in red_terms):
        return 'red'
    elif any(x in v for x in white_terms):
        return 'white'
    else:
        return 'other'

# Subtype classifications
full_red = {'shiraz', 'cabernet sauvignon', 'durif', 'malbec', 'mourvèdre', 'petit verdot'}
medium_red = {'merlot', 'grenache', 'tempranillo', 'sangiovese', 'montepulciano', 'cabernet franc', 'barbera', "nero d'avola"}
light_red = {'pinot noir', 'pinot meunier', 'gamay'}

light_white = {'sauvignon blanc', 'riesling', 'semillon', 'pinot gris', 'pinot grigio', 'grüner veltliner', 'marsanne', 'fiano', 'moscato'}
full_white = {'chardonnay', 'viognier', 'vermentino', 'verdelho', 'albariño', 'gewürztraminer', 'arneis'}

red_blends = {'red blend', 'bordeaux-style red blend'}
white_blends = {'white blend', 'bordeaux-style white blend'}

sparkling_types = {'champagne', 'prosecco'}
fortified_types = {'tokay', 'muscat', 'sherry', 'tawny', 'botrytis'}

def classify_wine_subtype(variety):
    # Handle NaN, None, or non-string values
    if pd.isna(variety) or not isinstance(variety, str):
        return 'unknown'
    
    v = variety.lower()
    if any(x in v for x in sparkling_types):
        if 'rosé' in v or 'rose' in v:
            return 'sparkling rosé'
        return 'sparkling'
    if any(x in v for x in fortified_types):
        return 'fortified'
    if 'rosé' in v or 'rose' in v:
        return 'rosé'
    if any(x in v for x in red_blends):
        return 'red blend'
    if any(x in v for x in white_blends):
        return 'white blend'
    if any(x in v for x in full_red):
        return 'full-bodied red'
    if any(x in v for x in medium_red):
        return 'medium-bodied red'
    if any(x in v for x in light_red):
        return 'light-bodied red'
    if any(x in v for x in light_white):
        return 'light-bodied white'
    if any(x in v for x in full_white):
        return 'full-bodied white'
    main = classify_wine_type_main(v)
    return main

def get_continent(country_name):
    try:
        country_code = pc.country_name_to_country_alpha2(country_name)
        continent_code = pc.country_alpha2_to_continent_code(country_code)
        return {
            'NA': 'North America',
            'SA': 'South America',
            'EU': 'Europe',
            'AS': 'Asia',
            'AF': 'Africa',
            'OC': 'Oceania',
        }[continent_code]
    except:
        return 'Other'

def engineer_batch_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for the batch dataset using the same transformations
    that were applied to the training data.
    
    Args:
        df: Batch dataset
        
    Returns:
        DataFrame with engineered features
    """
    # Remove index column if it exists
    df = remove_index_column(df)
    df = df.copy()

    # Points category
    df['points_category'] = df['points'].apply(classify_points)

    # Blend flag
    df['is_blend'] = df['variety'].str.contains("Blend", case=False)

    # Wine classification
    df['wine_type_main'] = df['variety'].apply(classify_wine_type_main)
    df['wine_subtype'] = df['variety'].apply(classify_wine_subtype)

    # Continent classification (standardize US/UK first)
    df['country_standardized'] = df['country'].replace({
        'US': 'United States',
        'England': 'United Kingdom'
    })
    df['continent'] = df['country_standardized'].apply(get_continent)

    return df

def create_one_hot_encoded_features(df):
    """
    Create one-hot encoded features from categorical columns.
    
    Args:
        df: DataFrame with feature engineered data
        
    Returns:
        DataFrame with one-hot encoded features
    """
    # Make a copy of the data
    df_encoded = df.copy()
    
    # Check if 'is_blend' column exists before trying to encode it
    if 'is_blend' in df_encoded.columns:
        df_encoded['is_blend_True'] = df_encoded['is_blend'].fillna(False).astype(int)
    else:
        # Create the column with default value if it doesn't exist
        df_encoded['is_blend_True'] = 0
    
    # List of categorical columns to one-hot encode
    categorical_columns = df_encoded.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    
    # Initialize the OneHotEncoder
    # Setting handle_unknown='ignore' to handle categories not seen during fit
    # drop='first' to avoid multicollinearity
    encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    
    # Fit and transform the categorical columns
    encoded_array = encoder.fit_transform(df_encoded[categorical_columns])
    
    # Get feature names from the encoder
    feature_names = encoder.get_feature_names_out(categorical_columns)
    
    # Create a dataframe with the encoded values
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=feature_names,
        index=df_encoded.index
    )
    
    # Concatenate the original dataframe with the encoded columns
    df_encoded = pd.concat([df_encoded.drop(columns=categorical_columns), encoded_df], axis=1)
    
    return df_encoded
