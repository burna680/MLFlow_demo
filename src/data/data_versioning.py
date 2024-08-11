import os
import pandas as pd

def version_data(X: pd.DataFrame, y: pd.DataFrame, data_path: str, version: str = 'v1') -> None:
    """
    Version data by saving it to csv files.

    Args:
        X (pd.DataFrame): The features DataFrame.
        y (pd.Series): The target Series.
        data_path (str): The path where the data will be saved.
        version (str, optional): The version of the data. Defaults to 'v1'.

    Returns:
        None
    """
    os.makedirs('data_versioning', exist_ok=True)

    # Define the paths for versioned data
    X_path = f'{data_path}/X_{version}.csv'
    y_path = f'{data_path}/y_{version}.csv'

    # Save the datasets
    X.to_csv(X_path, index=False)
    y.to_csv(y_path, index=False)
