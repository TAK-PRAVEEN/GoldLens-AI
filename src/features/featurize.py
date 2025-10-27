import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Ensure utils imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from utils.logging_config import setup_logging

logger = setup_logging()

RAW_DATA_PATH = Path("../GoldLens-AI/data/raw/gold_daily.csv")

def featurize(df):
    # df = df.sort_values("Date").reset_index(drop=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['DayOfWeek'] = df['Date'].dt.dayofweek

    # price derived
    df['Price_Range'] = df['High'] - df['Low']
    df['Avg_Price'] = (df['High'] + df['Low'])/2

    # lags
    for lag in [1,3,7,14,30]:
        df[f'Close_lag{lag}'] = df['Close'].shift(lag)

    # rolling statistics
    for w in [7,14,30,60,90]:
        df[f'MA_{w}'] = df['Close'].rolling(window=w).mean()
        df[f'STD_{w}'] = df['Close'].rolling(window=w).std()

    df['Return'] = df['Close'].pct_change()
    df = df.dropna().reset_index(drop=True)
    return df

# if __name__ == "__main__":
#     try:
#         logger.info(f"Loading raw gold price data from {RAW_DATA_PATH}")
#         df = pd.read_csv(RAW_DATA_PATH, parse_dates=["Date"])
#         logger.info(f"Raw data loaded with {len(df)} rows")

#         df_feat = featurize(df)
#         logger.info(f"Featurization complete. Output shape: {df_feat.shape}")

#         # Optionally save for future use
#         FEATURIZED_PATH = Path("../GoldLens-AI/data/processed/gold_daily_features.csv")
#         FEATURIZED_PATH.parent.mkdir(parents=True, exist_ok=True)
#         df_feat.to_csv(FEATURIZED_PATH, index=False)
#         logger.info(f"Saved featurized data to {FEATURIZED_PATH}")

#     except Exception as e:
#         logger.error(f"Error in loading or featurizing gold data: {e}")
#         raise
