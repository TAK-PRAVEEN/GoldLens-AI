import sys
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
from pathlib import Path

# Import logging setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from utils.logging_config import setup_logging

logger = setup_logging()

MODEL_DIR = Path("../GoldLens-AI/models")
DATA_PATH = Path("../GoldLens-AI/data/processed/gold_daily_features.csv")
window = 60

# Load scaler
scaler = joblib.load(MODEL_DIR / "scaler.pkl")

def load_all_models():
    """Load all models for prediction only (no compilation)."""
    models = {}
    for name in ['lstm', 'bilstm', 'gru']:
        path = MODEL_DIR / f"{name}.keras"
        if path.exists():
            models[name] = load_model(path, compile=False)
            logger.info(f"{name} model loaded for inference.")
        else:
            logger.error(f"Model file not found: {path}")
    return models

def batch_predict_ensemble(df):
    """
    Makes ensemble predictions using all models for a given test set.
    df: pandas DataFrame containing your processed features.

    Returns: dict with actuals and all predictions (inverse scaled).
    """
    close = df['Close'].values.reshape(-1, 1)
    scaled = scaler.transform(close)
    # Create windows for testing; skip train part for demo
    X = []
    y = []
    for i in range(window, len(scaled)):
        X.append(scaled[i-window:i])
        y.append(scaled[i])
    X = np.array(X).reshape((-1, window, 1))
    y = np.array(y)
    
    # Load models
    models = load_all_models()
    lstm = models['lstm']
    bilstm = models['bilstm']
    gru = models['gru']

    # Predictions
    pred_lstm = lstm.predict(X)
    pred_bilstm = bilstm.predict(X)
    pred_gru = gru.predict(X)
    
    # Weighted ensemble
    ensemble_pred = (pred_lstm + pred_bilstm + pred_gru)/3
    
    # Inverse scaling
    y_actual = scaler.inverse_transform(y.reshape(-1, 1))
    pred_lstm_actual = scaler.inverse_transform(pred_lstm)
    pred_bilstm_actual = scaler.inverse_transform(pred_bilstm)
    pred_gru_actual = scaler.inverse_transform(pred_gru)
    ensemble_actual = scaler.inverse_transform(ensemble_pred)

    logger.info("Batch ensemble predictions complete")
    return {
        'y_true': y_actual.flatten(),
        'lstm': pred_lstm_actual.flatten(),
        'bilstm': pred_bilstm_actual.flatten(),
        'gru': pred_gru_actual.flatten(),
        'ensemble': ensemble_actual.flatten()
    }

if __name__ == "__main__":
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, parse_dates=['Date'])
        results = batch_predict_ensemble(df)
        print("First 10 actuals:", results['y_true'][:10])
        print("First 10 ensemble preds:", results['ensemble'][:10])
        logger.info("Sample predictions written to stdout")
    else:
        logger.error(f"Feature data file not found: {DATA_PATH}")
