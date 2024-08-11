import pandas as pd
from ucimlrepo import fetch_ucirepo
from typing import Tuple

def gather_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetches the car evaluation dataset from the UCI ML repository and returns the features and targets.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: The features and targets of the car evaluation dataset.
    """
    car_evaluation = fetch_ucirepo(id=19)
    X: pd.DataFrame = car_evaluation.data.features
    y: pd.Series = car_evaluation.data.targets
    return X, y
