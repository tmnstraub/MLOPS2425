import numpy as np
import pandas as pd
import pytest
import warnings
from unittest.mock import Mock, patch
from PIL import Image
import io

warnings.filterwarnings("ignore", category=Warning)

from src.wine_project.pipelines.data_drift.nodes import (
    detect_univariate_drift,
    generate_drift_plot_image,
    _add_timestamp_from_index
)

def test_add_timestamp_from_index():
    """
    Test that the helper function correctly adds timestamp column from DataFrame index
    """
    df = pd.DataFrame({
        'points': [85, 90, 88],
        'price': [20.0, 30.0, 25.0]
    }, index=[10, 20, 30])
    
    result = _add_timestamp_from_index(df, 'record_id')
    
    assert 'record_id' in result.columns
    assert result['record_id'].tolist() == [10, 20, 30]
    assert len(result) == 3
    # Ensure original DataFrame is not modified
    assert 'record_id' not in df.columns

@pytest.mark.slow
def test_detect_univariate_drift():
    """
    Test that the drift detection node correctly identifies drift between datasets
    """
    reference_df, analysis_df = create_wine_dummy_data()
    
    params = {
        'features_to_monitor': ['points'],
        'timestamp_column': 'record_id',
        'chunk_size': 50
    }
    
    with patch('src.wine_project.pipelines.data_drift.nodes.nml.UnivariateDriftCalculator') as mock_calculator_class:
        # Setup mock
        mock_calculator = Mock()
        mock_calculator_class.return_value = mock_calculator
        mock_calculator.fit.return_value = mock_calculator
        
        # Mock drift results
        mock_drift_df = pd.DataFrame({
            'feature': ['points'],
            'chunk': [1],
            'kolmogorov_smirnov': [0.25],
            'jensen_shannon': [0.05],
            'drift_detected': [True]
        })
        
        mock_results = Mock()
        mock_results.to_df.return_value = mock_drift_df
        mock_calculator.calculate.return_value = mock_results
        
        # Run the drift detection
        drift_df, drift_results = detect_univariate_drift(reference_df, analysis_df, params)
        
        # Assertions
        assert isinstance(drift_df, pd.DataFrame)
        assert 'kolmogorov_smirnov' in drift_df.columns
        assert 'jensen_shannon' in drift_df.columns
        assert drift_results == mock_results
        
        # Verify NannyML was called correctly
        mock_calculator_class.assert_called_once_with(
            column_names=['points'],
            timestamp_column_name='record_id',
            continuous_methods=['kolmogorov_smirnov', 'jensen_shannon'],
            chunk_size=50
        )

def test_generate_drift_plot_image():
    """
    Test that the plot generation function returns a valid PIL Image
    """
    mock_drift_results = Mock()
    
    # Create a simple test image as bytes
    test_image = Image.new('RGB', (100, 50), color='blue')
    img_bytes = io.BytesIO()
    test_image.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    # Mock the plotly figure
    mock_figure = Mock()
    mock_figure.to_image.return_value = img_bytes.getvalue()
    mock_drift_results.plot.return_value = mock_figure
    
    # Run the function
    result_image = generate_drift_plot_image(mock_drift_results)
    
    # Assertions
    assert isinstance(result_image, Image.Image)
    assert result_image.mode == 'RGB'
    
    # Verify the mock was called correctly
    mock_drift_results.plot.assert_called_once_with(kind='drift')
    mock_figure.to_image.assert_called_once_with(
        format='png',
        width=1200,
        height=600,
        scale=2
    )

def create_wine_dummy_data():
    """
    Create dummy wine data for testing drift detection
    """
    np.random.seed(123)
    
    # Reference dataset
    reference_df = pd.DataFrame({
        'points': np.random.normal(88, 3, 100),
        'price': np.random.exponential(25, 100),
        'variety': np.random.choice(['Pinot Noir', 'Chardonnay', 'Merlot'], 100)
    })
    
    # Analysis dataset with slight drift
    analysis_df = pd.DataFrame({
        'points': np.random.normal(85, 4, 50),  # Lower mean, higher variance
        'price': np.random.exponential(25, 50),
        'variety': np.random.choice(['Pinot Noir', 'Chardonnay', 'Merlot'], 50)
    })
    
    # Ensure realistic ranges
    reference_df['points'] = np.clip(reference_df['points'], 80, 100)
    analysis_df['points'] = np.clip(analysis_df['points'], 80, 100)
    reference_df['price'] = np.abs(reference_df['price'])
    analysis_df['price'] = np.abs(analysis_df['price'])
    
    return reference_df, analysis_df
