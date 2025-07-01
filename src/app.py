"""
Module: app.py
Description: Main entry point for the machine learning pipeline.
"""

from loguru import logger
from flask import Flask, jsonify, request
import pandas as pd

from src.config import MODEL_PATH, FEATURES, CLASSES
from src.inference import load_model, predict


app = Flask(__name__)

MODEL = load_model(MODEL_PATH)


@app.route('/', methods=['GET'])
def start():
    """
    Default page
    """
    return "Welcome to the Machine Learning App!"


@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    """
    Health check endpoint to verify the application is running.
    """
    return jsonify({"status": "ok"})

# Example input: 
# {
#   "sepal_length": 5.1, 
#   "sepal_width": 3.5, 
#   "petal_length": 1.4, 
#   "petal_width": 0.2
# }

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """
    Endpoint to make predictions using the loaded model.
    Expects a JSON payload with feature values.
    """
    try:
        logger.info("Received prediction request")
        data = request.json
        logger.debug(f"Input data: {data}")
        df = pd.DataFrame(data, index=[0])
        df = df.reset_index(drop=True)
        df = df[FEATURES].copy()
        logger.debug(f"Processed DataFrame: {df.shape}")
        logger.info("Model prediction in progress...")
        prediction = predict(MODEL, df)
        logger.info("Model prediction completed!")
        result = {"prediction": CLASSES[prediction[0]]}
        logger.debug(f"Prediction result: {result}")
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

