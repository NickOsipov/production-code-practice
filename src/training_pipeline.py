import os
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from loguru import logger

from src.preprocessing import preprocessing_pipeline
from src.train import train_pipeline
from src.evaluate import evaluate_model
from src.inference import load_model
from src.config import MODEL_PATH, MODEL_PARAMS


def main():
    """
    Main function to run the machine learning pipeline.
    """
    logger.info("=========================================")
    logger.info("Starting the machine learning pipeline...")
    logger.info("Setting up model paths and params...")

    logger.info("Running preprocessing pipeline...")
    df_train, df_test = preprocessing_pipeline()
    
    logger.info("Training the model...")
    train_pipeline(RandomForestClassifier, MODEL_PARAMS, df_train, MODEL_PATH)

    logger.info("Model training completed. Saving the model...")
    model_loaded = load_model(MODEL_PATH)

    logger.info("Model loaded successfully. Evaluating the model...")
    accuracy_score = evaluate_model(model_loaded, df_test)
    logger.info(f"Model accuracy: {accuracy_score:.2f}")

    logger.info("Model evaluation completed.")
    logger.info("Pipeline finished successfully.")


if __name__ == "__main__":
    main()
    