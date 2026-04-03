"""
Model Architectures for Sales Forecasting.
Production-ready LSTM & GRU with configurable layers, dropout, and batch normalization.
"""
import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers


def build_lstm_model(input_shape, units=64, num_layers=2, dropout_rate=0.2,
                     learning_rate=0.001):
    """
    Build a production-grade stacked LSTM model.

    Args:
        input_shape: (lookback, num_features)
        units: LSTM hidden units
        num_layers: Number of LSTM layers
        dropout_rate: Dropout rate between layers
        learning_rate: Adam optimizer learning rate

    Returns:
        Compiled Keras model
    """
    model = keras.Sequential(name="LSTM_Sales_Forecaster")
    model.add(layers.Input(shape=input_shape))

    for i in range(num_layers):
        return_seq = (i < num_layers - 1)
        model.add(layers.LSTM(
            units,
            return_sequences=return_seq,
            name=f'lstm_{i+1}'
        ))
        model.add(layers.BatchNormalization(name=f'bn_{i+1}'))
        model.add(layers.Dropout(dropout_rate, name=f'dropout_{i+1}'))

    model.add(layers.Dense(32, activation='relu', name='dense_1'))
    model.add(layers.Dropout(dropout_rate, name='dropout_dense'))
    model.add(layers.Dense(16, activation='relu', name='dense_2'))
    model.add(layers.Dense(1, name='output'))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model


def build_gru_model(input_shape, units=64, num_layers=2, dropout_rate=0.2,
                    learning_rate=0.001):
    """
    Build a production-grade stacked GRU model.
    Mirrors the LSTM architecture depth and structure using GRU cells.

    Args:
        input_shape: (lookback, num_features)
        units: GRU hidden units
        num_layers: Number of GRU layers
        dropout_rate: Dropout rate between layers
        learning_rate: Adam optimizer learning rate

    Returns:
        Compiled Keras model
    """
    model = keras.Sequential(name="GRU_Sales_Forecaster")
    model.add(layers.Input(shape=input_shape))

    for i in range(num_layers):
        return_seq = (i < num_layers - 1)
        model.add(layers.GRU(
            units,
            return_sequences=return_seq,
            name=f'gru_{i+1}'
        ))
        model.add(layers.BatchNormalization(name=f'gru_bn_{i+1}'))
        model.add(layers.Dropout(dropout_rate, name=f'gru_dropout_{i+1}'))

    model.add(layers.Dense(32, activation='relu', name='gru_dense_1'))
    model.add(layers.Dropout(dropout_rate, name='gru_dropout_dense'))
    model.add(layers.Dense(16, activation='relu', name='gru_dense_2'))
    model.add(layers.Dense(1, name='gru_output'))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model


def get_callbacks(save_dir='models', patience=5, model_type='lstm'):
    """
    Get training callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau.

    Args:
        save_dir: Directory to save model checkpoints
        patience: Early stopping patience
        model_type: 'lstm' or 'gru' — determines checkpoint filename
    """
    os.makedirs(save_dir, exist_ok=True)

    model_filename = f'sales_{model_type}_model.keras'

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(save_dir, model_filename),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
    ]
    return callbacks


if __name__ == "__main__":
    print("=== LSTM Model ===")
    lstm = build_lstm_model(input_shape=(30, 25))
    lstm.summary()

    print("\n=== GRU Model ===")
    gru = build_gru_model(input_shape=(30, 25))
    gru.summary()
