"""
Punto de entrada del Módulo 1.
Ejecuta todo el pipeline: preprocesamiento → entrenamiento → visualizaciones.
"""

import os
import json

from src.preprocessing import (cargar_datos, construir_serie_demanda,
    rellenar_fechas_faltantes, agregar_features_temporales, preparar_datos_lstm)
from src.train import (LOOKBACK, EPOCHS, BATCH_SIZE, FIGURES_DIR,
    entrenar_ruta, guardar_metricas)
from src.visualization import (graficar_serie_historica, graficar_prediccion_vs_real,
    graficar_prediccion_futura, graficar_loss, graficar_descomposicion,
    graficar_demanda_por_dia_semana, tabla_metricas_comparativa)

os.makedirs('models', exist_ok=True)
os.makedirs('reports/figures', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# ── 1. Cargar y preprocesar ──────────────────────────────────────────────────
df_raw    = cargar_datos('data/raw/train_revised.csv')
demanda   = construir_serie_demanda(df_raw)
demanda   = rellenar_fechas_faltantes(demanda)
demanda   = agregar_features_temporales(demanda)

# ── 2. Gráficas exploratorias ────────────────────────────────────────────────
graficar_serie_historica(demanda)
graficar_demanda_por_dia_semana(demanda)

rutas = demanda['travel_from'].unique()

# ── 3. Entrenar un modelo por ruta ───────────────────────────────────────────
resultados = {}

for ruta in rutas:
    datos = preparar_datos_lstm(demanda, ruta, lookback=LOOKBACK)

    if len(datos['X_train']) < 10:
        print(f"[!] Ruta '{ruta}' omitida (pocos datos).")
        continue

    res = entrenar_ruta(ruta, datos, demanda)
    res['datos'] = datos
    resultados[ruta] = res

    # Gráficas por ruta
    graficar_prediccion_vs_real(
        res['metricas'], datos['fechas_test'], ruta
    )
    graficar_prediccion_futura(
        datos['df_ruta'], res['pred_futuro'], res['fechas_futuro'], ruta
    )
    graficar_loss(res['historia'], ruta)
    graficar_descomposicion(demanda, ruta)

# ── 4. Resumen global ────────────────────────────────────────────────────────
resumen = guardar_metricas(resultados)
tabla_metricas_comparativa(resumen)

print("\nMódulo 1 completado.")
print("   Revisa reports/figures/ para las gráficas")
print("   Revisa reports/predicciones_30_dias.csv para las predicciones")