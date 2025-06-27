import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import BaseEstimator

from inference import predict


def evaluate_model(model: BaseEstimator, df_test: pd.DataFrame) -> str:
    """
    Evaluate the performance of a machine learning model on a test DataFrame.

    Parameters
    ----------
    model : BaseEstimator
        The trained machine learning model to be evaluated.
    df_test : pd.DataFrame
        The DataFrame containing the test data, including features and target labels.
    """
    X_test = df_test.drop(columns=["target"])
    y_test = df_test["target"]

    predictions = predict(model, X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    # print("Classification Report:")
    # print(report)
    return accuracy
