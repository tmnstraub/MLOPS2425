import numpy as np
import pandas as pd
import pytest
import warnings
import mlflow,yaml
import logging

warnings.filterwarnings("ignore", category=Warning)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from src.wine_project.pipelines.model_selection.nodes import model_selection

logger = logging.getLogger(__name__)

@pytest.mark.slow
def test_model_selection():
    """
    Test that the model selection node returns a model with a score
    """
    # Create dummy data
    X_train = pd.DataFrame(np.random.rand(100, 5), columns=['feat1', 'feat2', 'feat3', 'feat4', 'feat5'])
    X_test = pd.DataFrame(np.random.rand(50, 5), columns=['feat1', 'feat2', 'feat3', 'feat4', 'feat5'])
    y_train = pd.DataFrame(np.random.randint(0, 2, size=100), columns=['target'])
    y_test = pd.DataFrame(np.random.randint(0, 2, size=50), columns=['target'])
    
    champion_dict = {'classifier': None, 'test_score': 0}
    
    champion_model = None
    
    parameters = {
        'hyperparameters': {
            'RandomForestClassifier': {'n_estimators': [10, 50, 100], 'max_depth': [3, 5, 10]},
            'GradientBoostingClassifier': {'learning_rate': [0.1, 0.01], 'n_estimators': [50, 100]},
        }
    }

    # Run the model selection node
    model = model_selection(X_train, X_test, y_train, y_test, champion_dict, champion_model, parameters)
    
    # Check that the returned value is a dictionary
    assert isinstance(model, RandomForestClassifier) or isinstance(model, GradientBoostingClassifier)
    assert isinstance(model.score(X_test, y_test), float)
