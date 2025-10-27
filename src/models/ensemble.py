import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import json
from pathlib import Path
import sys
import os

# Add your utility path for logging setup if outside cwd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from utils.logging_config import setup_logging

logger = setup_logging()

MODEL_DIR = Path("../GoldLens-AI/models")
metrics_path = MODEL_DIR / "metrics.json"

def ensemble():
    # Load best model configs (window sizes)
    with metrics_path.open() as f:
        metrics = json.load(f)

    model_meta = {
        'lstm': {
            'window': metrics['lstm']['cfg']['window'],
            'file': MODEL_DIR / "lstm_best.keras"
        },
        'bilstm': {
            'window': metrics['bilstm']['cfg']['window'],
            'file': MODEL_DIR / "bilstm_best.keras"
        },
        'gru': {
            'window': metrics['gru']['cfg']['window'],
            'file': MODEL_DIR / "gru_best.keras"
        }
    }

    # Load scaler
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    df = pd.read_csv("../GoldLens-AI/data/processed/gold_daily_features.csv", parse_dates=["Date"])
    close_values = df['Close'].values

    # Prepare prediction for each model
    predictions = []
    for name, meta in model_meta.items():
        window = meta['window']
        model_file = meta['file']
        if not model_file.exists():
            logger.warning(f"Model file missing: {model_file}")
            continue
        model = load_model(model_file)
        input_data = close_values[-window:]
        window_scaled = scaler.transform(np.array(input_data).reshape(-1, 1))
        pred = model.predict(window_scaled[np.newaxis])[0, 0]
        predictions.append(pred)
        logger.info(f"{name} prediction (window={window}): {pred:.2f}")

    if len(predictions) == 0:
        logger.error("No valid predictions! Check model files.")
    else:
        ensemble_pred = np.mean(predictions)
        logger.info(f"Ensemble mean prediction: {ensemble_pred:.2f}")
        print(f"Ensemble prediction: {ensemble_pred:.2f}")
