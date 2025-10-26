import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import joblib
import logging
import tensorflow.keras as keras

# Setup logging
from src.utils.logging_config import setup_logging
logger = setup_logging()

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_CSV = BASE_DIR / "data" / "processed" / "gold_daily_features.csv"
MODEL_DIR = BASE_DIR / "models"

# Define valid model names
VALID_MODEL_NAMES = [
    "lstm_best", "bilstm_best", "gru_best", "ensemble"
]

def load_data():
    """Load the gold price dataset with all features"""
    try:
        df = pd.read_csv(DATA_CSV, parse_dates=['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        logger.info(f"Loaded {len(df)} rows from {DATA_CSV}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def calculate_features(df, idx):
    row = df.iloc[idx]
    current_date = row['Date']
    features = {
        'Day': current_date.day,
        'Month': current_date.month,
        'Year': current_date.year,
        'DayOfWeek': current_date.dayofweek,
    }
    features['Open'] = row['Close']
    features['High'] = row['Close']
    features['Low'] = row['Close']
    features['Close'] = row['Close']
    features['Volume'] = row.get('Volume', df['Volume'].mean())
    features['Price_Range'] = 0
    features['Avg_Price'] = row['Close']
    features['Close_lag1'] = df['Close'].iloc[idx] if idx >= 0 else df['Close'].iloc[-1]
    features['Close_lag3'] = df['Close'].iloc[idx-2] if idx >= 2 else df['Close'].iloc[0]
    features['Close_lag7'] = df['Close'].iloc[idx-6] if idx >= 6 else df['Close'].iloc[0]
    features['Close_lag14'] = df['Close'].iloc[idx-13] if idx >= 13 else df['Close'].iloc[0]
    features['Close_lag30'] = df['Close'].iloc[idx-29] if idx >= 29 else df['Close'].iloc[0]
    close_series = df['Close'].iloc[max(0, idx-89):idx+1]
    features['MA_7'] = close_series.tail(7).mean() if len(close_series) >= 7 else close_series.mean()
    features['STD_7'] = close_series.tail(7).std() if len(close_series) >= 7 else 0
    features['MA_14'] = close_series.tail(14).mean() if len(close_series) >= 14 else close_series.mean()
    features['STD_14'] = close_series.tail(14).std() if len(close_series) >= 14 else 0
    features['MA_30'] = close_series.tail(30).mean() if len(close_series) >= 30 else close_series.mean()
    features['STD_30'] = close_series.tail(30).std() if len(close_series) >= 30 else 0
    features['MA_60'] = close_series.tail(60).mean() if len(close_series) >= 60 else close_series.mean()
    features['STD_60'] = close_series.tail(60).std() if len(close_series) >= 60 else 0
    features['MA_90'] = close_series.tail(90).mean() if len(close_series) >= 90 else close_series.mean()
    features['STD_90'] = close_series.tail(90).std() if len(close_series) >= 90 else 0
    if idx > 0:
        features['Return'] = (df['Close'].iloc[idx] - df['Close'].iloc[idx-1]) / df['Close'].iloc[idx-1]
    else:
        features['Return'] = 0
    return features

def load_model(model_name):
    """Load a trained model from the models directory (.keras format)"""
    try:
        if model_name not in VALID_MODEL_NAMES:
            raise ValueError(f"Invalid model name '{model_name}'. Choose from: {', '.join(VALID_MODEL_NAMES)}")
        model_path = MODEL_DIR / f"{model_name}.keras"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = keras.models.load_model(str(model_path))
        logger.info(f"Loaded model: {model_name}")
        return model
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        raise

def predict_future_prices(model_name, n_days):
    """
    Predict gold prices for the next n_days for all valid models (all expect windowed time series input).
    """
    try:
        logger.info(f"Starting prediction: model={model_name}, n_days={n_days}")
        model = load_model(model_name)
        df = load_data()
        last_idx = len(df) - 1
        last_date = df['Date'].iloc[last_idx]
        last_price = df['Close'].iloc[last_idx]
        predictions = []
        dates = []
        working_df = df.copy()

        # Load scaler (matching your model/data)
        scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        window = 60  # or your window size

        # Use only the 'Close' column for sequential models
        close_series = working_df['Close'].values

        for i in range(n_days):
            # Prepare the last `window` closes
            window_input = close_series[-window:]
            if len(window_input) < window:
                # Pad if needed (rare, only with very little data)
                window_input = np.pad(window_input, (window - len(window_input), 0), mode='edge')
            # Scale, then reshape to (1, window, 1)
            window_input_scaled = scaler.transform(window_input.reshape(-1, 1)).reshape(1, window, 1)
            pred_price_scaled = model.predict(window_input_scaled)[0][0]
            pred_price = scaler.inverse_transform([[pred_price_scaled]])[0][0]

            # Optional: Clamp outliers
            if pred_price < last_price * 0.5 or pred_price > last_price * 1.5:
                pred_price = last_price * (1 + np.random.uniform(-0.02, 0.02))

            predictions.append(float(pred_price))
            last_date += timedelta(days=1)
            dates.append(last_date.strftime('%Y-%m-%d'))
            # "Roll" window: append this prediction to close series
            close_series = np.append(close_series, pred_price)
            last_price = pred_price

        # Confidence intervals
        recent_std = df['Close'].tail(30).std()
        confidence_lower = [p - 1.96 * recent_std for p in predictions]
        confidence_upper = [p + 1.96 * recent_std for p in predictions]
        
        logger.info(f"Prediction completed: {len(predictions)} days")
        return {
            "success": True,
            "model": model_name,
            "n_days": n_days,
            "predictions": predictions,
            "dates": dates,
            "confidence_lower": confidence_lower,
            "confidence_upper": confidence_upper,
            "last_known_price": float(df['Close'].iloc[-1]),
            "last_known_date": df['Date'].iloc[-1].strftime('%Y-%m-%d')
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return {"error": str(e), "success": False}

def get_historical_data(days=120):
    """Get historical price data for chart display"""
    try:
        df = load_data()
        recent = df.tail(days)
        return {
            "dates": recent['Date'].dt.strftime('%Y-%m-%d').tolist(),
            "prices": recent['Close'].tolist()
        }
    except Exception as e:
        logger.error(f"Error getting historical data: {e}")
        return {"dates": [], "prices": []}

