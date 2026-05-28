"""
Arquitectura del modelo LSTM para predicción de demanda.
Basado en el enfoque del notebook de predicción de acciones,
adaptado para series de transporte.
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os


def construir_modelo_lstm(lookback=30, unidades_lstm1=32, unidades_lstm2=16, dropout=0.3):
    """
    Construye el modelo LSTM de dos capas.
    
    Args:
        lookback: tamaño de la ventana de entrada (timesteps)
        unidades_lstm1: neuronas en la primera capa LSTM
        unidades_lstm2: neuronas en la segunda capa LSTM
        dropout: tasa de dropout para regularización
    
    Returns:
        modelo compilado
    """
    modelo = Sequential([
        LSTM(
            unidades_lstm1,
            return_sequences=True,          # Devuelve secuencia para la siguiente LSTM
            input_shape=(lookback, 1)
        ),
        Dropout(dropout),

        LSTM(
            unidades_lstm2,
            return_sequences=False          # Solo el último estado para Dense
        ),
        Dropout(dropout),

        Dense(16, activation='relu'),
        Dense(1)                            # Salida: demanda normalizada
    ])

    modelo.compile(optimizer='adam', loss='mse', metrics=['mae'])
    modelo.summary()
    return modelo


def obtener_callbacks(ruta_checkpoint):
    """
    Callbacks para entrenamiento: detención temprana, guardado y lr adaptativo.
    """
    os.makedirs(os.path.dirname(ruta_checkpoint), exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,            # Para si no mejora en 15 épocas
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=ruta_checkpoint,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,             # Reduce lr a la mitad si se estanca
            patience=7,
            min_lr=1e-6,
            verbose=1
        )
    ]
    return callbacks