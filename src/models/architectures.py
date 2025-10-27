from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional

def make_lstm(input_shape, units=64, dropout=0.1):
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        LSTM(units // 2),
        Dropout(dropout),
        Dense(max(8, units // 4), activation='relu'),  # maintain at least 8 units
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def make_bilstm(input_shape, units=64, dropout=0.1):
    model = Sequential([
        Bidirectional(LSTM(units, return_sequences=True), input_shape=input_shape),
        Dropout(dropout),
        Bidirectional(LSTM(units // 2)),
        Dropout(dropout),
        Dense(max(8, units // 4), activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def make_gru(input_shape, units=64, dropout=0.1):
    model = Sequential([
        GRU(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        GRU(units // 2),
        Dropout(dropout),
        Dense(max(8, units // 4), activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
