import os

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from src.config import FEATURES


def load_data() -> pd.DataFrame:
    """
    Load the Iris dataset and split it into training and testing sets.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the Iris dataset with features and target labels.
    """
    iris = load_iris()
    X = iris.data
    y = iris.target
    df = pd.DataFrame(X, columns=FEATURES)
    df["target"] = y
    return df


def split_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42, stratify=None
) -> tuple:
    """
Split the dataset into training and testing sets.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the dataset to be split.
    test_size : float, optional
        The proportion of the dataset to include in the test split, by default 0.2
    random_state : int, optional
        Controls the shuffling applied to the data before applying the split, by default 42
    stratify : _type_, optional
        If not None, the data is split in a stratified fashion using this column, by default None

    Returns
    -------
    tuple
        A tuple containing the training and testing DataFrames.
    """
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=stratify)
    return df_train, df_test


def preprocessing_pipeline() -> tuple:
    """
    Run the preprocessing pipeline to load and split the Iris dataset.

    Returns
    -------
    tuple
        A tuple containing the training and testing DataFrames.
    """
    df = load_data()
    df_train, df_test = split_data(df, stratify=df["target"])
    return df_train, df_test
