import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent / 'api'))
sys.path.append(str(Path(__file__).resolve().parent / 'features'))
sys.path.append(str(Path(__file__).resolve().parent / 'ingest'))
sys.path.append(str(Path(__file__).resolve().parent / 'models'))
sys.path.append(str(Path(__file__).resolve().parent / 'utils'))

import logging

# Setup global logging if needed
from utils.logging_config import setup_logging
logger = setup_logging()

# 1. Ingest new data
from ingest import ingest  # you should have e.g. ingest.get_latest_gold_data()
# 2. Featurize data
from features import featurize  # e.g. featurize.make_features()
# 3. Train all models (existing train.py logic)
import models.train as train_main
# 4. Build/Save ensemble model if needed
import models.ensemble as ensemble_main

# Paths
DATA_DIR = Path("data")
RAW_CSV = DATA_DIR / "raw" / "gold_daily.csv"
FEATURES_CSV = DATA_DIR / "processed" / "gold_daily_features.csv"

def main():
    logger.info("=== NIGHTLY PIPELINE START ===")
    today = datetime.utcnow()
    yesterday = today - timedelta(days=1)

    # 1. Ingest Fresh Data
    logger.info("Ingesting gold data up to %s", yesterday.strftime('%Y-%m-%d'))
    ingest.get_latest_gold_data(end_date=yesterday.strftime('%Y-%m-%d')) # <== adapt this to your function

    # 2. Featurization
    logger.info("Featurizing data")
    featurize.make_features(input_path=RAW_CSV, output_path=FEATURES_CSV) # <== adapt as per your API

    # 3. Retrain Models
    logger.info("Retraining all models")
    # Call as in train.py's __main__ (adapt args as needed)
    train_main.train_all(
        processed_csv=FEATURES_CSV,
        window=60,
        epochs=60,
        batch_size=32
    )

    # 4. Build/Save Ensemble
    logger.info("Building/saving ensemble model")
    ensemble_main.save_ensemble_model() # assumes you have this from earlier

    logger.info("=== NIGHTLY PIPELINE COMPLETE ===")

if __name__ == "__main__":
    main()
