"""
Script de entrenamiento y evaluación del modelo LSTM
para predicción de demanda de transporte por ruta.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from preprocessing import cargar_datos, construir_serie_demanda, rellenar_fechas_faltantes, agregar_features_temporales, preparar_datos_lstm
from model import construir_modelo_lstm, obtener_callbacks
import random
import tensorflow as tf

os.environ['TF_DETERMINISTIC_OPS'] = '1'

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

EPOCHS = 100
BATCH_SIZE = 8
RUTAS_DIR = 'models'
FIGURES_DIR = 'reports/figures'


def calcular_metricas(y_real, y_pred, scaler):
    y_real_inv = scaler.inverse_transform(y_real.reshape(-1, 1)).flatten()
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(y_real_inv, y_pred_inv))
    mae  = mean_absolute_error(y_real_inv, y_pred_inv)
    r2   = r2_score(y_real_inv, y_pred_inv)

    mask = y_real_inv != 0
    mape = np.mean(np.abs((y_real_inv[mask] - y_pred_inv[mask]) / y_real_inv[mask])) * 100

    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2,
            'y_real': y_real_inv, 'y_pred': y_pred_inv}


def predecir_30_dias(modelo, ultima_ventana, scaler, lookback):
    ventana = ultima_ventana.copy().tolist()
    predicciones = []

    for _ in range(30):
        entrada = np.array(ventana[-lookback:]).reshape(1, lookback, 1)
        pred = modelo.predict(entrada, verbose=0)[0][0]
        predicciones.append(pred)
        ventana.append(pred)

    predicciones_inv = scaler.inverse_transform(
        np.array(predicciones).reshape(-1, 1)
    ).flatten()

    return predicciones_inv


def entrenar_ruta(ruta, datos, demanda_completa, lookback):
    print(f"\n{'='*50}")
    print(f"Entrenando ruta: {ruta}")
    print(f"{'='*50}")

    X_train = datos['X_train']
    y_train = datos['y_train']
    X_val   = datos['X_val']
    y_val   = datos['y_val']
    X_test  = datos['X_test']
    y_test  = datos['y_test']
    scaler  = datos['scaler']
    df_ruta = datos['df_ruta']

    modelo = construir_modelo_lstm(lookback=lookback)

    ruta_safe = ruta.replace(' ', '_')
    checkpoint_path = f"{RUTAS_DIR}/{ruta_safe}_best.keras"

    val_data = (X_val, y_val) if X_val is not None else None

    historia = modelo.fit(
        X_train, y_train,
        validation_data=val_data,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=obtener_callbacks(checkpoint_path),
        verbose=1
    )

    y_pred = modelo.predict(X_test, verbose=0).flatten()
    metricas = calcular_metricas(y_test, y_pred, scaler)

    print(f"\nMétricas [{ruta}]:")
    print(f"  RMSE: {metricas['RMSE']:.2f} pasajeros")
    print(f"  MAE:  {metricas['MAE']:.2f} pasajeros")
    print(f"  MAPE: {metricas['MAPE']:.2f}%")

    valores_norm = scaler.transform(df_ruta['demanda'].values.reshape(-1, 1)).flatten()
    ultima_ventana = valores_norm[-lookback:]
    pred_futuro = predecir_30_dias(modelo, ultima_ventana, scaler, lookback)

    ultima_fecha = df_ruta['travel_date'].max()
    fechas_futuro = pd.date_range(start=ultima_fecha + pd.Timedelta(days=1), periods=30)

    return {
        'modelo': modelo,
        'historia': historia,
        'metricas': metricas,
        'pred_futuro': pred_futuro,
        'fechas_futuro': fechas_futuro,
        'ruta_safe': ruta_safe
    }


def guardar_metricas(resultados_todas_rutas):
    """Guarda todas las métricas en JSON."""
    resumen = {}
    for ruta, res in resultados_todas_rutas.items():
        resumen[ruta] = {
            'RMSE': round(res['metricas']['RMSE'], 2),
            'MAE':  round(res['metricas']['MAE'], 2),
            'MAPE': round(res['metricas']['MAPE'], 2),
            'R2':   round(res['metricas']['R2'], 4),
        }
    with open('reports/metricas_evaluacion.json', 'w') as f:
        json.dump(resumen, f, indent=2)
    print("\nMétricas guardadas en reports/metricas_evaluacion.json")
    return resumen


if __name__ == '__main__':
    os.makedirs(RUTAS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    demanda = pd.read_csv('data/processed/demanda_diaria.csv', parse_dates=['travel_date'])

    # Configuración específica por ruta
    config_rutas = {
        'Kisii':       {'lookback': 14,  'fecha_desde': '2018-03-01'},
        'Migori':      {'lookback': 14, 'fecha_desde': None},
        'Homa Bay':    {'lookback': 14, 'fecha_desde': None},
        'Otras Rutas': {'lookback': 21, 'fecha_desde': None},
    }

    resultados = {}

    for ruta, cfg in config_rutas.items():
        print(f"\n{'='*50}\nEntrenando ruta: {ruta}\n{'='*50}")

        # Filtrar periodo si aplica
        demanda_ruta = demanda.copy()
        if cfg['fecha_desde']:
            demanda_ruta = demanda_ruta[
                (demanda_ruta['travel_from'] != ruta) |
                (demanda_ruta['travel_date'] >= cfg['fecha_desde'])
            ]

        datos = preparar_datos_lstm(demanda_ruta, ruta, lookback=cfg['lookback'])

        if len(datos['X_train']) < 10:
            print(f"[!] Ruta '{ruta}' omitida (pocos datos).")
            continue

        res = entrenar_ruta(ruta, datos, demanda_ruta, lookback=cfg['lookback'])
        res['datos'] = datos
        resultados[ruta] = res

    guardar_metricas(resultados)

    predicciones_futuras = []
    for ruta, res in resultados.items():
        for fecha, pred in zip(res['fechas_futuro'], res['pred_futuro']):
            predicciones_futuras.append({
                'ruta': ruta,
                'fecha': str(fecha.date()),
                'demanda_predicha': max(0, round(pred))
            })

    df_pred = pd.DataFrame(predicciones_futuras)
    df_pred.to_csv('reports/predicciones_30_dias.csv', index=False)
    print("\nPredicciones guardadas en reports/predicciones_30_dias.csv")
    print(df_pred)