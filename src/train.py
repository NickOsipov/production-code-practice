import joblib

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator


def train_model(model: BaseEstimator, model_params: dict, df_train: pd.DataFrame) -> BaseEstimator:
    """
    Train a machine learning model using the provided training DataFrame.

    Parameters
    ----------
    model : BaseEstimator
        The machine learning model to be trained.
    model_params : dict
        A dictionary containing the parameters for the model.
    df_train : pd.DataFrame
        The DataFrame containing the training data.

    Returns
    -------
    BaseEstimator
        The trained machine learning model.
    """
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    
    model_instance = model(**model_params)
    model_instance.fit(X_train, y_train)
    
    return model_instance


def save_model(model: BaseEstimator, model_path: str) -> None:
    """
    Save the trained machine learning model to a file.

    Parameters
    ----------
    model : BaseEstimator
        The trained machine learning model to be saved.
    model_path : str
        The file path where the model will be saved.
    """
    joblib.dump(model, model_path)


def train_pipeline(model: BaseEstimator, model_params: dict, df_train: pd.DataFrame, model_path: str) -> None:
    """
    Run the training pipeline to train and save the machine learning model.

    Parameters
    ----------
    model : BaseEstimator
        The machine learning model to be trained.
    model_params : dict
        A dictionary containing the parameters for the model.
    df_train : pd.DataFrame
        The DataFrame containing the training data.
    model_path : str
        The file path where the trained model will be saved.
    """
    trained_model = train_model(model, model_params, df_train)
    save_model(trained_model, model_path)