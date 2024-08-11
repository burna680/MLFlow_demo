from pandas import DataFrame

def analyze_data(X: DataFrame, y: DataFrame) -> None:
    """
    Analyze the given DataFrame and Series.

    Parameters:
    ----------
    X: pandas.DataFrame
        The feature DataFrame.
    y: pandas.Series
        The target Series.

    Returns:
    -------
    None
    """
    print(X.columns, y.columns)
    print(X.info(), y.describe(include='all'))
    print(X.head(), y.head())

    for col in X.columns:
        print(X[col].value_counts())
    print(y.value_counts())

    print(X.isnull().sum())
