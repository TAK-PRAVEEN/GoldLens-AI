import sys
import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Ensure project path imports correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from utils.logging_config import setup_logging

# Initialize centralized logger
logger = setup_logging()

# Define data directory
DATA_DIR = Path("../GoldLens-AI/data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def fetch_gold_data(ticker="GC=F", start_date="2000-01-01", end_date=None):
    """
    Fetch historical gold futures data (GC=F) using yfinance.
    Uses yf.Ticker().history() to ensure consistent and detailed OHLCV data.
    """
    try:
        logger.info(f"Fetching gold futures for {ticker} from {start_date} to {end_date or 'today'}")
        gold = yf.Ticker(ticker)
        gold_data = gold.history(start=start_date, end=end_date, auto_adjust=False)

        if gold_data.empty:
            raise RuntimeError(f"No data returned for {ticker}. Check ticker symbol or network connection.")

        gold_data.reset_index(inplace=True)
        gold_data = gold_data[["Date", "Open", "High", "Low", "Close", "Volume"]]
        logger.info(f"Fetched {len(gold_data)} rows of data for {ticker}")
        return gold_data
    except Exception as e:
        logger.error(f"Error fetching gold data for {ticker}: {e}")
        raise

def append_daily(ticker="GC=F", start_date="2000-01-01"):
    """
    Append latest daily gold data into 'gold_daily.csv'.
    Automatically updates only if new data is available.
    """
    dest = DATA_DIR / "gold_daily.csv"

    try:
        if dest.exists():
            existing = pd.read_csv(dest, parse_dates=["Date"])
            last_date = existing["Date"].max().date()
            today = datetime.utcnow().date()

            if last_date >= today:
                logger.info("No new gold price data to update.")
                return

            logger.info(f"Existing data until {last_date}. Fetching updates from {last_date + timedelta(days=1)}...")
            gold = yf.Ticker(ticker)
            new_data = gold.history(
                start=(last_date + timedelta(days=1)).strftime("%Y-%m-%d"),
                end=(today + timedelta(days=1)).strftime("%Y-%m-%d")
            )

            if new_data.empty:
                logger.info("No new rows returned from Yahoo Finance.")
                return

            new_data.reset_index(inplace=True)
            new_data = new_data[["Date", "Open", "High", "Low", "Close", "Volume"]]
            combined = pd.concat([existing, new_data], ignore_index=True)

            combined.to_csv(dest, index=False)
            logger.info(f"Appended {len(new_data)} new rows. Total rows: {len(combined)}")

        else:
            logger.info("Dataset not found. Downloading full history from Yahoo Finance...")
            df = fetch_gold_data(ticker=ticker, start_date=start_date)
            df.to_csv(dest, index=False)
            logger.info(f"Saved initial dataset ({len(df)} rows) to {dest}")

    except Exception as e:
        logger.error(f"Error updating gold data: {e}")
        raise

if __name__ == "__main__":
    logger.info("Starting gold data fetch pipeline...")
    append_daily()
    logger.info("Gold data fetch pipeline completed successfully.")
