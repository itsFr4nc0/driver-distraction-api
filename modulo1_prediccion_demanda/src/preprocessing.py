"""
Preprocesamiento del dataset de demanda de transporte de Nairobi.
Convierte registros individuales de pasajeros en series de tiempo
de demanda diaria por ruta.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import pickle


def cargar_datos(ruta_archivo):
    """Carga y hace limpieza básica del CSV crudo."""
    df = pd.read_csv(ruta_archivo)

    # Parsear fecha: formato DD-MM-YY
    df['travel_date'] = pd.to_datetime(df['travel_date'], format='%d-%m-%y')

    # Eliminar duplicados exactos
    df = df.drop_duplicates()

    # Eliminar filas con fecha nula
    df = df.dropna(subset=['travel_date', 'travel_from'])

    print(f"Registros cargados: {len(df)}")
    print(f"Rango de fechas: {df['travel_date'].min()} → {df['travel_date'].max()}")
    print(f"Rutas únicas (origen): {df['travel_from'].unique()}")
    return df


def construir_serie_demanda(df):
    """
    Agrega la demanda diaria por ruta (travel_from).
    Cada fila del resultado = pasajeros totales que viajaron
    desde esa ciudad hacia Nairobi en esa fecha.
    """
    demanda = (
        df.groupby(['travel_date', 'travel_from'])
        .size()
        .reset_index(name='demanda')
        .sort_values(['travel_from', 'travel_date'])
    )
    return demanda


def rellenar_fechas_faltantes(demanda):
    """
    Crea un índice de fechas completo por ruta y rellena los
    días sin viajes con demanda 0.
    """
    rutas = demanda['travel_from'].unique()
    fecha_min = demanda['travel_date'].min()
    fecha_max = demanda['travel_date'].max()
    rango_fechas = pd.date_range(start=fecha_min, end=fecha_max, freq='D')

    bloques = []
    for ruta in rutas:
        df_ruta = demanda[demanda['travel_from'] == ruta].set_index('travel_date')
        df_ruta = df_ruta.reindex(rango_fechas, fill_value=0)
        df_ruta.index.name = 'travel_date'
        df_ruta['travel_from'] = ruta
        df_ruta = df_ruta.reset_index()[['travel_date', 'travel_from', 'demanda']]
        bloques.append(df_ruta)

    demanda_completa = pd.concat(bloques, ignore_index=True)
    return demanda_completa


def agregar_features_temporales(df):
    """Agrega columnas de día de la semana, mes, etc."""
    df = df.copy()
    df['dia_semana'] = df['travel_date'].dt.dayofweek      # 0=lunes
    df['mes'] = df['travel_date'].dt.month
    df['dia_mes'] = df['travel_date'].dt.day
    df['es_fin_semana'] = (df['dia_semana'] >= 5).astype(int)
    return df


def crear_ventanas(serie, lookback=30):
    """
    Transforma una serie 1D en pares (X, y) de ventana deslizante.
    
    Args:
        serie: array 1D normalizado
        lookback: cuántos días pasados usar para predecir el siguiente
    
    Returns:
        X: shape (n_samples, lookback, 1)
        y: shape (n_samples,)
    """
    X, y = [], []
    for i in range(lookback, len(serie)):
        X.append(serie[i - lookback:i])
        y.append(serie[i])
    return np.array(X).reshape(-1, lookback, 1), np.array(y)


def preparar_datos_lstm(demanda_completa, ruta, lookback=30, test_ratio=0.15, val_ratio=0.15):
    """
    Pipeline completo para una ruta específica:
    normaliza, crea ventanas y hace el split train/val/test.
    
    Returns:
        dict con X_train, X_val, X_test, y_train, y_val, y_test, scaler, fechas_test
    """
    df_ruta = demanda_completa[demanda_completa['travel_from'] == ruta].copy()
    df_ruta = df_ruta.sort_values('travel_date').reset_index(drop=True)

    valores = df_ruta['demanda'].values.reshape(-1, 1)

    # Normalizar
    scaler = MinMaxScaler(feature_range=(0, 1))
    valores_norm = scaler.fit_transform(valores).flatten()

    # Calcular índices de corte
    n = len(valores_norm)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_test - n_val

    train = valores_norm[:n_train]
    val = valores_norm[n_train:n_train + n_val]
    # Para test incluimos los últimos `lookback` de val como contexto
    test = valores_norm[n_train + n_val - lookback:]

    X_train, y_train = crear_ventanas(train, lookback)
    X_val, y_val = crear_ventanas(val, lookback) if len(val) > lookback else (None, None)
    X_test, y_test = crear_ventanas(test, lookback)

    # Fechas del periodo de test (para graficar)
    fechas_test = df_ruta['travel_date'].values[n_train + n_val:]

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'scaler': scaler,
        'fechas_test': fechas_test,
        'df_ruta': df_ruta
    }


def guardar_procesado(demanda_completa, ruta_salida='data/processed/demanda_diaria.csv'):
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    demanda_completa.to_csv(ruta_salida, index=False)
    print(f"Guardado en: {ruta_salida}")

def agrupar_rutas(demanda_completa):
    """
    Agrupa rutas en 4 categorías:
    - Kisii, Migori, Homa Bay → se mantienen individuales
    - Todo lo demás → 'Otras Rutas'
    """
    RUTAS_PRINCIPALES = ['Kisii', 'Migori', 'Homa Bay']
    
    def asignar_grupo(ruta):
        return ruta if ruta in RUTAS_PRINCIPALES else 'Otras Rutas'
    
    demanda_agrupada = demanda_completa.copy()
    demanda_agrupada['travel_from'] = demanda_agrupada['travel_from'].apply(asignar_grupo)
    
    # Re-agrupar sumando la demanda
    demanda_agrupada = (
        demanda_agrupada
        .groupby(['travel_date', 'travel_from'])['demanda']
        .sum()
        .reset_index()
    )
    
    # Re-agregar features temporales
    demanda_agrupada = agregar_features_temporales(demanda_agrupada)
    
    return demanda_agrupada

if __name__ == '__main__':
    df_raw = cargar_datos('data/raw/train_revised.csv')
    demanda = construir_serie_demanda(df_raw)
    demanda = rellenar_fechas_faltantes(demanda)
    demanda = agregar_features_temporales(demanda)
    
    # Aplicar agrupación
    demanda_agrupada = agrupar_rutas(demanda)
    guardar_procesado(demanda_agrupada, 'data/processed/demanda_diaria.csv')

    print("\nDemanda diaria agrupada (resumen):")
    print(demanda_agrupada.groupby('travel_from')['demanda'].describe())