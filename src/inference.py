import joblib

import pandas as pd

from sklearn.base import BaseEstimator


def load_model(model_path: str) -> BaseEstimator:
    """
    Load a machine learning model from a file.

    Parameters
    ----------
    model_path : str
        The file path where the model is saved.

    Returns
    -------
    BaseEstimator
        The loaded machine learning model.
    """
    model = joblib.load(model_path)
    return model


def predict(model: BaseEstimator, X: pd.DataFrame) -> pd.Series:
    """
    Make predictions using the loaded machine learning model.

    Parameters
    ----------
    model : BaseEstimator
        The loaded machine learning model.
    X : pd.DataFrame
        The DataFrame containing the features for prediction.

    Returns
    -------
    pd.Series
        A Series containing the predicted values.
    """
    predictions = model.predict(X)
    return pd.Series(predictions)
