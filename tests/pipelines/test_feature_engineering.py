import pandas as pd
import pytest
from src.wine_project.pipelines.feature_engineering.nodes import engineer_features

def test_engineer_features():
    """Test the feature engineering function with a small dataset."""
    # Create test input data
    test_data = pd.DataFrame({
        'description': ['A red wine with fruity notes', 'White wine with citrus flavor'],
        'variety': ['Cabernet Sauvignon', 'Chardonnay'],
        'points': [90, 85],
        'price': [45.0, 20.0]
        # Add other columns that your function expects
    })
    
    # Run the feature engineering function
    result = engineer_features(test_data)
    
    # Assert the results meet expectations
    assert 'quality_category' in result.columns  # Check new features exist
    assert 'wine_type' in result.columns
    
    # Check specific values
    assert result.loc[0, 'wine_type'] == 'Red'  # First row should be red wine
    assert result.loc[1, 'wine_type'] == 'White'  # Second row should be white wine