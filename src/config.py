"""
Module: config.py
Description: Configuration settings for the machine learning pipeline.
"""

import os

MODEL_PATH = os.path.join("models", "model.joblib")
MODEL_PARAMS = {"n_estimators": 100, "max_depth": 5, "random_state": 42}

FEATURES = [
    # iris dataset features
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]
CLASSES = [
    # iris dataset classes
    "setosa",
    "versicolor",
    "virginica",
]