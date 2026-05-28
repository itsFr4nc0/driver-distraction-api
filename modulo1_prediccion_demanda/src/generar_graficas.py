"""
Genera todas las gráficas del Módulo 1.
Ejecutar después de train.py
"""

import pandas as pd
import numpy as np
import os
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error

from preprocessing import preparar_datos_lstm
from visualization import (graficar_serie_historica, graficar_prediccion_vs_real,
    graficar_prediccion_futura, graficar_descomposicion,
    graficar_demanda_por_dia_semana, tabla_metricas_comparativa)
from model import construir_modelo_lstm

config_rutas = {
    'Kisii':       {'lookback': 14, 'fecha_desde': '2018-03-01'},
    'Migori':      {'lookback': 14, 'fecha_desde': None},
    'Homa Bay':    {'lookback': 14, 'fecha_desde': None},
    'Otras Rutas': {'lookback': 21, 'fecha_desde': None},
}

# Cargar datos procesados
demanda = pd.read_csv('data/processed/demanda_diaria.csv', parse_dates=['travel_date'])

# Cargar métricas y predicciones
with open('reports/metricas_evaluacion.json') as f:
    resumen_metricas = json.load(f)

df_pred_futuro = pd.read_csv('reports/predicciones_30_dias.csv', parse_dates=['fecha'])

# Gráficas generales
graficar_serie_historica(demanda)
graficar_demanda_por_dia_semana(demanda)
tabla_metricas_comparativa(resumen_metricas)

# Gráficas por ruta
for ruta, cfg in config_rutas.items():
    print(f"\nGenerando gráficas para: {ruta}")
    ruta_safe = ruta.replace(' ', '_')
    lookback = cfg['lookback']

    # Filtrar periodo igual que en train
    demanda_ruta = demanda.copy()
    if cfg['fecha_desde']:
        demanda_ruta = demanda_ruta[
            (demanda_ruta['travel_from'] != ruta) |
            (demanda_ruta['travel_date'] >= cfg['fecha_desde'])
        ]

    datos = preparar_datos_lstm(demanda_ruta, ruta, lookback=lookback)

    # Cargar modelo con pesos entrenados
    modelo = construir_modelo_lstm(lookback=lookback)
    modelo.load_weights(f'models/{ruta_safe}_best.keras')

    # Predicciones en test
    y_pred_norm = modelo.predict(datos['X_test'], verbose=0).flatten()
    scaler = datos['scaler']

    y_real_inv = scaler.inverse_transform(datos['y_test'].reshape(-1, 1)).flatten()
    y_pred_inv = scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(y_real_inv, y_pred_inv))
    mae  = mean_absolute_error(y_real_inv, y_pred_inv)
    mask = y_real_inv != 0
    mape = np.mean(np.abs((y_real_inv[mask] - y_pred_inv[mask]) / y_real_inv[mask])) * 100 if mask.sum() > 0 else float('nan')

    metricas = {
        'RMSE': rmse, 'MAE': mae, 'MAPE': mape,
        'y_real': y_real_inv, 'y_pred': y_pred_inv
    }

    # Predicciones futuras
    pred_futuro_ruta = df_pred_futuro[df_pred_futuro['ruta'] == ruta]
    fechas_futuro = pred_futuro_ruta['fecha'].values
    pred_futuro   = pred_futuro_ruta['demanda_predicha'].values

    graficar_prediccion_vs_real(metricas, datos['fechas_test'], ruta)
    graficar_prediccion_futura(datos['df_ruta'], pred_futuro, fechas_futuro, ruta)
    graficar_descomposicion(demanda, ruta)

print("\n✅ Todas las gráficas generadas en reports/figures/")