import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from MLOPS2425.pipelines.feature_selection.nodes import correlation_feature_selection
from MLOPS2425.pipelines.feature_selection.nodes import feature_selection_rfe

def test_correlation_selection_drops_less_important_feature():
    """
    Test that given two highly correlated features, the one with the
    lower correlation to the target variable is dropped.
    """
    # Arrange: Create sample data
    # 'feat_A' and 'feat_B' are highly correlated (0.98)
    # 'feat_A' has a higher correlation with 'target' (0.9) than 'feat_B' (0.6)
    # 'noise' has low correlation with everything.
    X_train = pd.DataFrame({
        "feat_A": [1, 2, 3, 4, 5],
        "feat_B": [1.1, 2.1, 3.1, 4.1, 5.1],
        "noise":  [5, 2, 8, 3, 1],
    })
    y_train = pd.Series([1, 2, 3, 4, 5], name="target") # Perfectly correlated with feat_A

    params = {"correlation_threshold": 0.95}

    # Act: Run the node function
    selected_features = correlation_feature_selection(X_train, y_train, params)

    # Assert: Check the result
    # We expect 'feat_B' to be dropped.
    assert "feat_B" not in selected_features
    assert "feat_A" in selected_features
    assert "noise" in selected_features
    assert len(selected_features) == 2


def test_rfe_selection_keeps_predictive_features():
    """Test that RFE correctly identifies and keeps predictive features."""
    # Arrange: Create sample data
    # 'important_feat' is perfectly predictive of the target.
    # 'noise_1' and 'noise_2' are random.
    X_train = pd.DataFrame({
        "important_feat": [10, 20, 10, 20, 10],
        "noise_1":        [1, 5, 2, 8, 3],
        "noise_2":        [9, 2, 4, 7, 5],
    })
    y_train = pd.Series([1, 0, 1, 0, 1])

    # Create a simple mock model. It doesn't have to be a RandomForest.
    # We configure RFE to select only 1 feature.
    mock_model = LogisticRegression()
    params = {"n_features_to_select": 1} # RFE can take this param

    # Act: Run the node function
    # NOTE: We'll modify the node slightly to accept n_features_to_select
    selected_features = feature_selection_rfe(X_train, y_train, mock_model, params)

    # Assert: Check that only the important feature was selected
    assert selected_features == ["important_feat"]


def test_intersect_features_logic():
    """Test the intersection and fallback logic."""
    rfe = ["a", "b", "c"]
    corr = ["b", "c", "d"]

    # Test intersection
    result = intersect_features(rfe, corr)
    assert set(result) == {"b", "c"}

    # Test fallback (no intersection)
    corr_no_overlap = ["d", "e", "f"]
    result_fallback = intersect_features(rfe, corr_no_overlap)
    assert result_fallback == rfe # Should return the RFE list
