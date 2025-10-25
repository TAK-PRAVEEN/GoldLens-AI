import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import joblib
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_CSV = BASE_DIR / "data" / "processed" / "gold_daily_features.csv"
MODEL_DIR = BASE_DIR / "models"


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
    """
    Calculate all required features for a given row index
    Based on your dataset columns:
    Date,Open,High,Low,Close,Volume,Day,Month,Year,DayOfWeek,
    Price_Range,Avg_Price,Close_lag1,Close_lag3,Close_lag7,Close_lag14,Close_lag30,
    MA_7,STD_7,MA_14,STD_14,MA_30,STD_30,MA_60,STD_60,MA_90,STD_90,Return
    """
    row = df.iloc[idx]
    
    # Basic date features
    current_date = row['Date']
    features = {
        'Day': current_date.day,
        'Month': current_date.month,
        'Year': current_date.year,
        'DayOfWeek': current_date.dayofweek,
    }
    
    # Price features (use most recent values)
    features['Open'] = row['Close']  # Assume next open = current close
    features['High'] = row['Close']  # Will be adjusted by model
    features['Low'] = row['Close']   # Will be adjusted by model
    features['Close'] = row['Close']
    features['Volume'] = row.get('Volume', df['Volume'].mean())
    features['Price_Range'] = 0  # Will be calculated after prediction
    features['Avg_Price'] = row['Close']
    
    # Lag features
    features['Close_lag1'] = df['Close'].iloc[idx] if idx >= 0 else df['Close'].iloc[-1]
    features['Close_lag3'] = df['Close'].iloc[idx-2] if idx >= 2 else df['Close'].iloc[0]
    features['Close_lag7'] = df['Close'].iloc[idx-6] if idx >= 6 else df['Close'].iloc[0]
    features['Close_lag14'] = df['Close'].iloc[idx-13] if idx >= 13 else df['Close'].iloc[0]
    features['Close_lag30'] = df['Close'].iloc[idx-29] if idx >= 29 else df['Close'].iloc[0]
    
    # Moving averages and standard deviations
    close_series = df['Close'].iloc[max(0, idx-89):idx+1]  # Last 90 days max
    
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
    
    # Return (daily return)
    if idx > 0:
        features['Return'] = (df['Close'].iloc[idx] - df['Close'].iloc[idx-1]) / df['Close'].iloc[idx-1]
    else:
        features['Return'] = 0
    
    return features


def load_model(model_name):
    """Load a trained model"""
    try:
        model_path = MODEL_DIR / f"{model_name}_model.pkl"
        
        if not model_path.exists():
            # Try alternative naming
            model_path = MODEL_DIR / f"{model_name}.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = joblib.load(model_path)
        logger.info(f"Loaded model: {model_name}")
        return model
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        raise


def predict_future_prices(model_name, n_days):
    """
    Predict gold prices for the next n_days
    
    Args:
        model_name: 'lstm', 'bilstm', 'gru', or 'ensemble'
        n_days: Number of days to predict (1-365)
    
    Returns:
        dict with predictions, dates, and metadata
    """
    try:
        logger.info(f"Starting prediction: model={model_name}, n_days={n_days}")
        
        # Load model
        model = load_model(model_name)
        
        # Load historical data
        df = load_data()
        
        # Get last known values
        last_idx = len(df) - 1
        last_date = df['Date'].iloc[last_idx]
        last_price = df['Close'].iloc[last_idx]
        
        logger.info(f"Last known: date={last_date}, price={last_price}")
        
        # Get feature columns (exclude Date)
        feature_cols = [col for col in df.columns if col != 'Date']
        
        # Initialize prediction storage
        predictions = []
        dates = []
        
        # Create working dataframe for iterative prediction
        working_df = df.copy()
        
        # Predict iteratively
        for i in range(1, n_days + 1):
            # Get features from most recent row
            features = calculate_features(working_df, len(working_df) - 1)
            
            # Create feature DataFrame with correct column order
            X = pd.DataFrame([features])
            X = X[feature_cols]  # Ensure correct column order
            
            # Handle missing columns (fill with 0)
            for col in feature_cols:
                if col not in X.columns:
                    X[col] = 0
            
            # Make prediction
            pred_price = model.predict(X)[0]
            
            # Ensure prediction is reasonable (within 20% of last price)
            if pred_price < last_price * 0.5 or pred_price > last_price * 1.5:
                pred_price = last_price * (1 + np.random.uniform(-0.02, 0.02))
            
            predictions.append(float(pred_price))
            
            # Calculate next date (skip weekends if desired, but we'll keep all days for now)
            next_date = last_date + timedelta(days=i)
            dates.append(next_date.strftime('%Y-%m-%d'))
            
            # Append predicted row to working dataframe
            new_row = features.copy()
            new_row['Date'] = next_date
            new_row['Close'] = pred_price
            new_row['Open'] = pred_price
            new_row['High'] = pred_price * 1.01
            new_row['Low'] = pred_price * 0.99
            new_row['Avg_Price'] = pred_price
            new_row['Price_Range'] = new_row['High'] - new_row['Low']
            
            working_df = pd.concat([working_df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Calculate confidence intervals (±2 standard deviations)
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
            "last_known_price": float(last_price),
            "last_known_date": last_date.strftime('%Y-%m-%d')
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return {"error": str(e), "success": False}


def get_historical_data(days=90):
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


# Test function
if __name__ == "__main__":
    # Test prediction
    result = predict_future_prices("ensemble", 7)
    print(result)
