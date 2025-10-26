import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Average

def build_ensemble_model(model_paths, input_shape):
    """
    Build a Keras ensemble model using the averaged outputs of base models.
    Args:
        model_paths: list of file paths to base model .keras files
        input_shape: tuple, shape of a single input sample (e.g., (60, 1))
    Returns:
        Keras Model instance
    """
    # Load all base models
    base_models = [load_model(path, compile=False) for path in model_paths]
    
    # Set them to non-trainable
    for m in base_models:
        m.trainable = False
    
    # Define input
    ensemble_input = Input(shape=input_shape, name="ensemble_input")
    
    # Model outputs
    model_outputs = [m(ensemble_input) for m in base_models]
    
    # Average the outputs
    avg = Average()(model_outputs)
    
    # Build ensemble model
    ensemble_model = Model(inputs=ensemble_input, outputs=avg, name="ensemble")
    return ensemble_model

def save_ensemble_model():
    """
    Build and save the ensemble model as models/ensemble.keras
    """
    from pathlib import Path

    MODEL_DIR = Path("../GoldLens-AI/models")
    window = 60  # Should match your window size

    # Paths to best checkpoints of your models
    model_paths = [
        MODEL_DIR / "lstm_best.keras",
        MODEL_DIR / "bilstm_best.keras",
        MODEL_DIR / "gru_best.keras",
    ]
    model_paths = [str(p) for p in model_paths]

    ensemble = build_ensemble_model(model_paths, input_shape=(window, 1))
    # Save ensemble model
    outpath = MODEL_DIR / "ensemble.keras"
    ensemble.save(outpath)
    print(f"Ensemble model saved to {outpath}")

if __name__ == "__main__":
    save_ensemble_model()
