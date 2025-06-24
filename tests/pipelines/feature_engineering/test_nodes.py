import pandas as pd
import pytest
import datetime

# Import directly from src
from src.wine_project.pipelines.feature_engineering.nodes import engineer_features

def test_engineer_features():
    """Test the feature engineering function."""
    # Create test input data with all required columns
    test_data = pd.DataFrame({
        'country': ['US', 'France'],
        'points': [90, 85],
        'price': [45.0, 20.0],
        'province': ['California', 'Burgundy'],
        'region_1': ['Napa Valley', 'Chablis'],
        'taster_name': ['John Doe', 'Jane Smith'],
        'variety': ['Cabernet Sauvignon', 'Chardonnay'],
        'index': [1, 2],
        'datetime': [datetime.datetime.now(), datetime.datetime.now()]
            })
    
    # Run the feature engineering function
    result = engineer_features(test_data)
    

    # At the end of your test_engineer_features function:
    # Assert the expected outputs
    assert 'wine_type_main' in result.columns  # Changed from 'wine_type'
    assert 'wine_subtype' in result.columns
    assert 'country_standardized' in result.columns
    assert 'continent' in result.columns
    assert 'points_category' in result.columns
    assert 'is_blend' in result.columns