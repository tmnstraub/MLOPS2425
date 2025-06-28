import pandas as pd
import numpy as np

from src.wine_project.pipelines.train_val_split.nodes import split_data

def test_split_data():
    """Test the split_data function.
    """

    # Create a sample DataFrame
    data = {
        'Id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Feature1': [1, 40, 50, 40, 50,40,20,90,10,10],
        'Feature2': [100, 10, 30, 1, 0,3,5,7,8, 9],
        'Target': [1, 0, 1, 0, 1, 1,0,1,1,0 ],
        'index' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'datetime' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
    df = pd.DataFrame(data)

    # Define the parameters
    parameters = {
        'target_column': 'Target',
        'random_state': 2021,
        'test_fraction': 0.2
    }

    # Call the split_data function
    X_train, X_test, y_train, y_test, columns_list = split_data(df, parameters)


    # Assert the existence of the datasets
    assert X_train is not None
    assert X_test is not None
    assert y_train is not None
    assert y_test is not None


    # Assert the shapes of the resulting datasets
    assert X_train.shape == (8, 4)
    assert X_test.shape == (2, 4)
    assert y_train.shape == (8,)
    assert y_test.shape == (2,)




