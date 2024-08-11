from sklearn.model_selection import train_test_split
import category_encoders as ce
import pandas as pd
from typing import Tuple

def prepare_data( X: pd.DataFrame, y: pd.DataFrame, test_size: float = 0.33,random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare training and testing data for ML model.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.33.
        random_state (int, optional): Controls the randomness of train_test_split. Defaults to 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            X_train (pd.DataFrame): Training features.
            X_test (pd.DataFrame): Testing features.
            y_train (pd.Series): Training target variable.
            y_test (pd.Series): Testing target variable.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    encoder = ce.OrdinalEncoder(cols=list(X.columns))
    X_train = encoder.fit_transform(X_train)
    X_test = encoder.transform(X_test)

    return X_train, X_test, y_train, y_test
