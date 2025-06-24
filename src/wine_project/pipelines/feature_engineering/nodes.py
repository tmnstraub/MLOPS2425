import pandas as pd
import pycountry_convert as pc

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
medium_red = {'merlot', 'grenache', 'tempranillo', 'sangiovese', 'montepulciano', 'cabernet franc', 'barbera', "nero d’avola"}
light_red = {'pinot noir', 'pinot meunier', 'gamay'}

light_white = {'sauvignon blanc', 'riesling', 'semillon', 'pinot gris', 'pinot grigio', 'grüner veltliner', 'marsanne', 'fiano', 'moscato'}
full_white = {'chardonnay', 'viognier', 'vermentino', 'verdelho', 'albariño', 'gewürztraminer', 'arneis'}

red_blends = {'red blend', 'bordeaux-style red blend'}
white_blends = {'white blend', 'bordeaux-style white blend'}

sparkling_types = {'champagne', 'prosecco'}
fortified_types = {'tokay', 'muscat', 'sherry', 'tawny', 'botrytis'}

def classify_wine_subtype(variety):
    v = str(variety).lower()
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

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
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
