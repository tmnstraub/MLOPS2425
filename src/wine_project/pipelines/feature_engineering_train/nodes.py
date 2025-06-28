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
    'dolcetto', 'pinotage',
    # Added terms based on analysis
    'cabernet franc', 'petit verdot', 'grenache', 'shiraz', 'barbera',
    'garnacha', 'petitesirah', 'gamay', 'corvina', 'aglianico',
    # Regional indicators for red wine
    'bordeaux', 'burgundy', 'beaujolais', 'barolo', 'chianti', 'rioja'
]

white_terms = [
    'white', 'pinot bianco', 'pinot blanc', 'grüner veltliner', 'sauvignon blanc',
    'riesling', 'pinot gris', 'pinot grigio', 'melon', 'vermentino', 'sémillon',
    'fiano', 'alvarinho', 'friulano', 'nerello', 'greco', 'grillo',
    # Added terms based on analysis
    'viognier', 'gewürztraminer', 'chenin blanc', 'malvasia', 'verdejo',
    'trebbiano', 'moschofilero', 'moscato',
    # Regional indicators for white wine
    'chablis', 'sancerre', 'soave'
]

fortified_terms = ['port', 'sherry', 'tokay', 'muscat', 'tawny', 'botrytis']

# Additional indicators for better pattern matching
red_indicators = ['malbec', 'cab', 'noir', 'shiraz', 'syrah', 'zinfandel', 'merlot']
white_indicators = ['blanc', 'sauv', 'chard', 'riesling', 'grigio', 'gris', 'semillon']

def classify_wine_type_main(variety):
    """
    Classify wine variety into main type categories using normalized string matching
    for better accuracy.
    """
    # Handle NaN, None, or non-string values
    if pd.isna(variety) or not isinstance(variety, str):
        return 'other'
    
    # Normalize the input string - lowercase and remove spaces
    v = variety.lower().replace(' ', '')
    
    # Normalize the term lists to remove spaces
    normalized_red_terms = [term.lower().replace(' ', '') for term in red_terms]
    normalized_white_terms = [term.lower().replace(' ', '') for term in white_terms]
    normalized_fortified = [term.lower().replace(' ', '') for term in fortified_terms]
    
    # Check for sparkling wines
    sparkling_patterns = ['champagne', 'sparkling', 'prosecco', 'spumante', 'cava']
    if any(pattern.lower().replace(' ', '') in v for pattern in sparkling_patterns):
        return 'sparkling'
    
    # Check for rosé wines
    rose_patterns = ['rosé', 'rose', 'rosado', 'rosato']
    if any(pattern.lower().replace(' ', '') in v for pattern in rose_patterns):
        return 'rosé'
    
    # Check for fortified wines
    if any(term.replace(' ', '') in v for term in normalized_fortified):
        return 'fortified'
    
    # Check for red wines using normalized terms
    if any(term in v for term in normalized_red_terms):
        return 'red'
    
    # Check for white wines using normalized terms
    if any(term in v for term in normalized_white_terms):
        return 'white'
    
    # Check for additional indicators
    if any(ind in v for ind in red_indicators):
        return 'red'
    
    if any(ind in v for ind in white_indicators):
        return 'white'
    
    # Check for blends
    if 'blend' in v:
        return 'red blend'  # Most blends are red by default
    
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
    """
    Classify wine variety into more specific subtypes with improved pattern matching.
    """
    # Handle NaN, None, or non-string values
    if pd.isna(variety) or not isinstance(variety, str):
        return 'other'
    
    # Normalize the input
    v = variety.lower().replace(' ', '')
    
    # Normalize subtype sets to handle string matching better
    normalized_full_red = {x.lower().replace(' ', '') for x in full_red}
    normalized_medium_red = {x.lower().replace(' ', '') for x in medium_red}
    normalized_light_red = {x.lower().replace(' ', '') for x in light_red}
    normalized_light_white = {x.lower().replace(' ', '') for x in light_white}
    normalized_full_white = {x.lower().replace(' ', '') for x in full_white}
    normalized_red_blends = {x.lower().replace(' ', '') for x in red_blends}
    normalized_white_blends = {x.lower().replace(' ', '') for x in white_blends}
    normalized_sparkling = {x.lower().replace(' ', '') for x in sparkling_types}
    normalized_fortified = {x.lower().replace(' ', '') for x in fortified_types}
    
    # Sparkling
    if any(x in v for x in normalized_sparkling):
        if any(x in v for x in ['rosé', 'rose', 'rosado', 'rosato']):
            return 'sparkling rosé'
        return 'sparkling'
    
    # Fortified/dessert
    if any(x in v for x in normalized_fortified):
        return 'fortified'
    
    # Rosé
    if any(x in v for x in ['rosé', 'rose', 'rosado', 'rosato']):
        return 'rosé'
    
    # Blends
    if any(x in v for x in normalized_red_blends):
        return 'red blend'
    if any(x in v for x in normalized_white_blends):
        return 'white blend'
    
    # Red subtypes
    for term in normalized_full_red:
        if term in v:
            return 'full-bodied red'
    
    for term in normalized_medium_red:
        if term in v:
            return 'medium-bodied red'
    
    for term in normalized_light_red:
        if term in v:
            return 'light-bodied red'
    
    # White subtypes
    for term in normalized_light_white:
        if term in v:
            return 'light-bodied white'
    
    for term in normalized_full_white:
        if term in v:
            return 'full-bodied white'
    
    # Fallback to main classification with improved pattern matching
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

def engineer_train_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for the training dataset without filtering.
    
    Args:
        df: Training dataset
        
    Returns:
        DataFrame with all engineered features
    """
    # Remove index column if it exists
    df = remove_index_column(df)
    df = df.copy()

    # Points category
    df['points_category'] = df['points'].apply(classify_points)

    # Blend flag - ensure the variety column exists and is string type
    if 'variety' in df.columns:
        # Convert to string first to handle any non-string values
        df['variety'] = df['variety'].astype(str)
        # Create is_blend feature
        df['is_blend'] = df['variety'].str.contains("blend", case=False)
    else:
        # Create a default is_blend column if variety doesn't exist
        df['is_blend'] = False

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

def select_train_features(df: pd.DataFrame, features_to_drop: list = None) -> pd.DataFrame:
    """
    Select features from the engineered dataset by manually specifying which ones to drop.
    
    Args:
        df: Feature-engineered dataset
        features_to_drop: List of column names to exclude from the dataset
        
    Returns:
        DataFrame with selected features
    """
    # Create a copy of the dataframe
    df_selected = df.copy()
    
    # If features_to_drop is not provided or empty, return the original dataframe
    if not features_to_drop:
        print("No features specified to drop. Keeping all features.")
        return df_selected
    
    # Print the features that will be dropped for debugging
    print(f"Features specified to drop: {features_to_drop}")
    
    # Filter the features_to_drop list to only include columns that actually exist in the dataframe
    valid_features_to_drop = [col for col in features_to_drop if col in df_selected.columns]
    
    # Check if any specified features don't exist in the dataframe
    if len(valid_features_to_drop) < len(features_to_drop):
        missing_features = set(features_to_drop) - set(valid_features_to_drop)
        print(f"Warning: The following features don't exist in the dataframe: {missing_features}")
    
    # Drop the specified features
    if valid_features_to_drop:
        print(f"Dropping {len(valid_features_to_drop)} features: {valid_features_to_drop}")
        df_selected = df_selected.drop(columns=valid_features_to_drop)
    
    return df_selected

def create_one_hot_encoded_features(df: pd.DataFrame) -> pd.DataFrame:
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
        # Only drop the original is_blend column if it exists
        df_encoded = df_encoded.drop(columns=['is_blend'])
    else:
        # Create the column with default value if it doesn't exist
        df_encoded['is_blend_True'] = 0
    
    # List of categorical columns to one-hot encode
    categorical_columns = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()

    # Initialize the OneHotEncoder
    # Setting handle_unknown='ignore' to handle categories not seen during fit
    # drop='first' to avoid multicollinearity
    encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    
    # Fit and transform the categorical columns
    if categorical_columns:  # Only proceed if there are categorical columns
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
    
    # Check if 'is_blend' column exists before attempting to use it
    if 'is_blend' in df_encoded.columns:
        # One-hot encode the boolean column, handling NaN values
        # Fill NaN values with False (0) before converting to int
        df_encoded['is_blend_True'] = df_encoded['is_blend'].fillna(False).astype(int)
        
        # Drop the original is_blend column
        df_encoded = df_encoded.drop(columns=['is_blend'])
    
    return df_encoded
    
