from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def train_model( X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int = 100, random_state: int = 0) -> RandomForestClassifier:
    """
    Train a random forest classifier on the given training data.

    Args:
        X_train (pd.DataFrame): The feature DataFrame.
        y_train (pd.Series): The target Series.
        n_estimators (int, optional): The number of trees in the forest. Defaults to 100.
        random_state (int, optional): The random state for reproducibility. Defaults to 0.

    Returns:
        RandomForestClassifier: The trained random forest classifier.
    """
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train, y_train)
    return clf
