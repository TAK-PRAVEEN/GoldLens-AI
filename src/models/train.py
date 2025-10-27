import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import sys
import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import itertools

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

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

def train_and_evaluate(model_fn, X_train, y_train, X_val, y_val, epochs, batch_size, model_name, params):
    model = model_fn(**params)
    ckpt = MODEL_DIR / f"{model_name}_best.keras"
    cb = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ModelCheckpoint(str(ckpt), save_best_only=True, monitor='val_loss')
    ]
    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=epochs, batch_size=batch_size,
              callbacks=cb, verbose=0)
    pred = model.predict(X_val)
    return model, pred

def train_all(processed_csv, search=True, epochs=60, batch_size=32):
    try:
        logger.info("Reading data from %s", processed_csv)
        df = pd.read_csv(processed_csv, parse_dates=["Date"])
        close = df['Close'].ffill().values.reshape(-1, 1)

        # Set hyperparameter space
        if search:
            param_grid = {
                "window": [30, 45, 60],
                "units": [32, 64, 128],
                "dropout": [0.1, 0.2],
            }
        else:
            param_grid = {
                "window": [60],
                "units": [128],
                "dropout": [0.2],
            }
        grid = list(itertools.product(param_grid['window'], param_grid['units'], param_grid['dropout']))

        best_results = {}
        all_results = {}
        final_scaler = None

        # Find best config for each model
        for model_name, model_fn in zip(
                ['lstm', 'bilstm', 'gru'],
                [make_lstm, make_bilstm, make_gru]
        ):
            best_score = float('inf')
            best_model = None
            best_scaler = None
            best_cfg = None
            best_pred = None
            y_val_true_best = None

            logger.info(f"Grid search for {model_name} ... {len(grid)} configs")
            for window, units, drop in grid:
                scaler = MinMaxScaler()
                scaled = scaler.fit_transform(close)
                X, y = windowed_dataset(scaled.flatten(), window)
                if len(X) == 0:
                    logger.warning(f"Window size {window} too large for data length {len(scaled)}")
                    continue
                X = X.reshape((X.shape[0], X.shape[1], 1))
                split = int(0.8 * len(X))
                X_train, X_val = X[:split], X[split:]
                y_train, y_val = y[:split], y[split:]
                params = {'input_shape': (X_train.shape[1], 1), 'units': units, 'dropout': drop}
                model, pred = train_and_evaluate(
                    lambda input_shape, units, dropout: model_fn(input_shape=input_shape, units=units, dropout=dropout),
                    X_train, y_train, X_val, y_val,
                    epochs=epochs, batch_size=batch_size,
                    model_name=f"{model_name}_temp", params=params
                )
                pred_inv = scaler.inverse_transform(pred)
                y_val_true = scaler.inverse_transform(y_val.reshape(-1, 1))
                val_mae = mean_absolute_error(y_val_true, pred_inv)
                logger.info(f"{model_name} window={window} units={units} dropout={drop} val_MAE={val_mae:.4f}")
                if val_mae < best_score:
                    best_score = val_mae
                    best_cfg = {"window": window, "units": units, "dropout": drop}
                    best_model = model
                    best_scaler = scaler
                    best_pred = pred_inv
                    y_val_true_best = y_val_true
            # Save best model and its scaler
            best_model.save(MODEL_DIR / f"{model_name}_best.keras")
            joblib.dump(best_scaler, MODEL_DIR / f"{model_name}_scaler.pkl")
            # Calculate metrics
            mae = mean_absolute_error(y_val_true_best, best_pred)
            rmse = mean_squared_error(y_val_true_best, best_pred)
            r2 = r2_score(y_val_true_best, best_pred)
            best_results[model_name] = dict(cfg=best_cfg, mae=float(mae), rmse=float(rmse), r2=float(r2))
            logger.info(f"Best {model_name}: {best_results[model_name]}")
            final_scaler = best_scaler

        # Ensemble using saved models!
        # Use the best "window" parameter found (from last) -- could be different for each, but we'll use the last cfg's window.
        best_window = best_cfg["window"]
        X_full, y_full = windowed_dataset(final_scaler.fit_transform(close).flatten(), best_window)
        X_full = X_full.reshape((X_full.shape[0], X_full.shape[1], 1))
        split = int(0.8 * len(X_full))
        X_val_f, y_val_f = X_full[split:], y_full[split:]
        preds = []
        for model_name in ['lstm', 'bilstm', 'gru']:
            mdl = load_model(MODEL_DIR / f"{model_name}_best.keras")
            preds.append(mdl.predict(X_val_f))
        ensemble_pred = np.mean(preds, axis=0)
        ensemble_inv = final_scaler.inverse_transform(ensemble_pred)
        y_true_ensemble = final_scaler.inverse_transform(y_val_f.reshape(-1, 1))
        ensemble_mae = mean_absolute_error(y_true_ensemble, ensemble_inv)
        ensemble_rmse = mean_squared_error(y_true_ensemble, ensemble_inv)
        ensemble_r2 = r2_score(y_true_ensemble, ensemble_inv)
        best_results['ensemble'] = dict(mae=float(ensemble_mae), rmse=float(ensemble_rmse), r2=float(ensemble_r2))
        # Save last-used scaler as main scaler for predictions
        joblib.dump(final_scaler, MODEL_DIR / "scaler.pkl")
        (MODEL_DIR / "metrics.json").write_text(json.dumps(best_results, indent=2))
        logger.info("Grid search training complete. Metrics: %s", json.dumps(best_results, indent=2))
        return best_results
    except Exception as e:
        logger.exception(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    test_csv = ROOT / "data/processed/gold_daily_features.csv"
    if test_csv.exists():
        logger.info("Running test training routine with %s", test_csv)
        try:
            # By default, enables grid search
            metrics = train_all(test_csv, search=True)
            logger.info("Test training finished successfully: %s", metrics)
        except Exception as e:
            logger.error("Test case failed: %s", e)
    else:
        logger.error("Test CSV not found: %s", test_csv)
