import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from utils.logging_config import setup_logging
from models.architectures import make_lstm, make_bilstm, make_gru

logger = setup_logging()

ROOT = Path("../GoldLens-AI/")
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def windowed_dataset(series, window_size):
    X, y = [], []
    for i in range(window_size, len(series)):
        X.append(series[i - window_size:i])
        y.append(series[i])
    return np.array(X), np.array(y)

def train_all(processed_csv, window=60, epochs=60, batch_size=32):
    try:
        logger.info("Reading data from %s", processed_csv)
        df = pd.read_csv(processed_csv, parse_dates=["Date"])
        # Ensure target is Close and drop any missing values in the main features
        close = df['Close'].fillna(method='ffill').values.reshape(-1, 1)  # forward fill for any missing
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(close)

        X, y = windowed_dataset(scaled.flatten(), window)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        models = {
            'lstm': make_lstm((X_train.shape[1], 1)),
            'bilstm': make_bilstm((X_train.shape[1], 1)),
            'gru': make_gru((X_train.shape[1], 1)),
        }
        results = {}

        # Store predictions for ensemble
        preds_dict = {}

        for name, model in models.items():
            logger.info("Training %s model", name)
            ckpt = MODEL_DIR / f"{name}_best.keras"
            cb = [
                EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
                ModelCheckpoint(str(ckpt), save_best_only=True, monitor='val_loss')
            ]
            model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs, batch_size=batch_size,
                callbacks=cb, verbose=1
            )
            # Predict
            pred = model.predict(X_test)
            preds_dict[name] = pred

            # Inverse scaling for metrics
            pred_inv = scaler.inverse_transform(pred)
            y_true = scaler.inverse_transform(y_test.reshape(-1, 1))
            mae = mean_absolute_error(y_true, pred_inv)
            rmse = mean_squared_error(y_true, pred_inv)
            r2 = r2_score(y_true, pred_inv)

            results[name] = dict(mae=float(mae), rmse=float(rmse), r2=float(r2))
            model.save(MODEL_DIR / f"{name}.keras")
            logger.info("%s metrics: MAE=%.4f RMSE=%.4f R2=%.4f", name, mae, rmse, r2)
            # Log actual vs predicted sample
            logger.info(f"Sample actual vs {name} preds:\nActual: {y_true[:5].flatten()}\nPredicted: {pred_inv[:5].flatten()}")

        # Weighted Ensemble
        ensemble_pred = (preds_dict['lstm'] + preds_dict['bilstm'] + preds_dict['gru'])/3
        ensemble_inv = scaler.inverse_transform(ensemble_pred)
        y_true = scaler.inverse_transform(y_test.reshape(-1, 1))
        ensemble_mae = mean_absolute_error(y_true, ensemble_inv)
        ensemble_rmse = mean_squared_error(y_true, ensemble_inv)
        ensemble_r2 = r2_score(y_true, ensemble_inv)
        results['ensemble'] = dict(mae=float(ensemble_mae), rmse=float(ensemble_rmse), r2=float(ensemble_r2))

        logger.info("Ensemble metrics: MAE=%.4f RMSE=%.4f R2=%.4f", ensemble_mae, ensemble_rmse, ensemble_r2)
        logger.info(f"Sample actual vs ensemble preds:\nActual: {y_true[:5].flatten()}\nEnsemble: {ensemble_inv[:5].flatten()}")
        # Save scaler and all metrics
        joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
        (MODEL_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
        logger.info("Training complete. Metrics: %s", results)
        return results

    except Exception as e:
        logger.exception(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    test_csv = ROOT / "data/processed/gold_daily_features.csv"
    if test_csv.exists():
        logger.info("Running test training routine with %s", test_csv)
        try:
            metrics = train_all(test_csv)  # Use default params for test
            logger.info("Test training finished successfully: %s", metrics)
        except Exception as e:
            logger.error("Test case failed: %s", e)
    else:
        logger.error("Test CSV not found: %s", test_csv)
