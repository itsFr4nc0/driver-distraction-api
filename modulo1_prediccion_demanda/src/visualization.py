"""
Generación de todas las gráficas del módulo 1.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose
import os


FIGURES_DIR = 'reports/figures'
os.makedirs(FIGURES_DIR, exist_ok=True)


def graficar_serie_historica(demanda_completa):
    """Demanda diaria histórica por ruta en un solo gráfico."""
    rutas = demanda_completa['travel_from'].unique()
    fig, axes = plt.subplots(len(rutas), 1, figsize=(14, 3 * len(rutas)), sharex=True)

    if len(rutas) == 1:
        axes = [axes]

    for ax, ruta in zip(axes, rutas):
        df_r = demanda_completa[demanda_completa['travel_from'] == ruta]
        ax.plot(df_r['travel_date'], df_r['demanda'], linewidth=1, color='steelblue')
        ax.set_title(f'Ruta: {ruta}', fontsize=11, fontweight='bold')
        ax.set_ylabel('Pasajeros')
        ax.grid(alpha=0.3)

    plt.suptitle('Demanda Histórica por Ruta', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/01_series_historicas.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Guardada: 01_series_historicas.png")


def graficar_prediccion_vs_real(metricas, fechas_test, ruta):
    """Predicción vs real en el periodo de test."""
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(fechas_test, metricas['y_real'], label='Real', color='steelblue', linewidth=1.5)
    ax.plot(fechas_test, metricas['y_pred'], label='Predicción LSTM',
            color='tomato', linewidth=1.5, linestyle='--')

    ax.set_title(f'Predicción vs Real — {ruta}\n'
                 f"RMSE={metricas['RMSE']:.1f}  MAE={metricas['MAE']:.1f}  MAPE={metricas['MAPE']:.1f}%",
                 fontsize=11)
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Pasajeros')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=30)
    plt.tight_layout()

    ruta_safe = ruta.replace(' ', '_')
    plt.savefig(f'{FIGURES_DIR}/02_pred_vs_real_{ruta_safe}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Guardada: 02_pred_vs_real_{ruta_safe}.png")


def graficar_prediccion_futura(df_ruta, pred_futuro, fechas_futuro, ruta):
    """Muestra histórico + predicción de los próximos 30 días."""
    fig, ax = plt.subplots(figsize=(14, 5))

    # Últimos 90 días históricos para contexto visual
    df_reciente = df_ruta.tail(90)
    ax.plot(df_reciente['travel_date'], df_reciente['demanda'],
            color='steelblue', linewidth=1.5, label='Histórico (últimos 90 días)')

    # Predicción futura
    ax.plot(fechas_futuro, pred_futuro,
            color='darkorange', linewidth=2, linestyle='--',
            marker='o', markersize=4, label='Predicción 30 días')

    # Línea vertical separando presente y futuro
    ax.axvline(x=df_ruta['travel_date'].max(), color='gray',
               linestyle=':', linewidth=1.5, label='Hoy')

    # Banda de incertidumbre aproximada (±10%)
    ax.fill_between(fechas_futuro,
                    pred_futuro * 0.90, pred_futuro * 1.10,
                    alpha=0.2, color='darkorange', label='Margen ±10%')

    ax.set_title(f'Predicción 30 Días Futuros — {ruta}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Pasajeros')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.xticks(rotation=30)
    plt.tight_layout()

    ruta_safe = ruta.replace(' ', '_')
    plt.savefig(f'{FIGURES_DIR}/03_prediccion_futura_{ruta_safe}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Guardada: 03_prediccion_futura_{ruta_safe}.png")


def graficar_loss(historia, ruta):
    """Curva de pérdida durante el entrenamiento."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(historia.history['loss'], label='Train Loss', color='steelblue')
    if 'val_loss' in historia.history:
        ax.plot(historia.history['val_loss'], label='Val Loss', color='tomato')
    ax.set_title(f'Pérdida de Entrenamiento — {ruta}')
    ax.set_xlabel('Época')
    ax.set_ylabel('MSE')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    ruta_safe = ruta.replace(' ', '_')
    plt.savefig(f'{FIGURES_DIR}/04_loss_{ruta_safe}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Guardada: 04_loss_{ruta_safe}.png")


def graficar_descomposicion(demanda_completa, ruta, period=7):
    """Descomposición estacional STL: tendencia, estacionalidad, residuo."""
    df_r = demanda_completa[demanda_completa['travel_from'] == ruta].copy()
    df_r = df_r.set_index('travel_date')['demanda']

    # Necesita al menos 2 periodos completos
    if len(df_r) < 2 * period:
        print(f"[!] {ruta}: datos insuficientes para descomposición.")
        return

    try:
        resultado = seasonal_decompose(df_r, model='additive', period=period)
        fig = resultado.plot()
        fig.set_size_inches(12, 8)
        fig.suptitle(f'Descomposición Estacional — {ruta}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        ruta_safe = ruta.replace(' ', '_')
        fig.savefig(f'{FIGURES_DIR}/05_descomposicion_{ruta_safe}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Guardada: 05_descomposicion_{ruta_safe}.png")
    except Exception as e:
        print(f"[!] Error en descomposición de {ruta}: {e}")


def graficar_demanda_por_dia_semana(demanda_completa):
    """Boxplot de demanda según día de la semana por ruta."""
    dias = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
    rutas = demanda_completa['travel_from'].unique()

    fig, axes = plt.subplots(1, len(rutas), figsize=(4 * len(rutas), 5), sharey=False)
    if len(rutas) == 1:
        axes = [axes]

    for ax, ruta in zip(axes, rutas):
        df_r = demanda_completa[demanda_completa['travel_from'] == ruta]
        grupos = [df_r[df_r['dia_semana'] == i]['demanda'].values for i in range(7)]
        ax.boxplot(grupos, labels=dias)
        ax.set_title(ruta, fontsize=10, fontweight='bold')
        ax.set_ylabel('Pasajeros')
        ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Demanda por Día de la Semana', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/06_demanda_dia_semana.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Guardada: 06_demanda_dia_semana.png")


def tabla_metricas_comparativa(resumen_metricas):
    """Tabla visual comparando métricas de todas las rutas."""
    df_m = pd.DataFrame(resumen_metricas).T.reset_index()
    # Solo mostrar RMSE, MAE, MAPE — omitir R2
    df_m = df_m[['index', 'RMSE', 'MAE', 'MAPE']]
    df_m.columns = ['Ruta', 'RMSE', 'MAE', 'MAPE (%)']

    fig, ax = plt.subplots(figsize=(8, len(df_m) * 0.6 + 1.5))
    ax.axis('off')
    tabla = ax.table(
        cellText=df_m.values,
        colLabels=df_m.columns,
        cellLoc='center',
        loc='center'
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(11)
    tabla.scale(1.2, 1.8)
    ax.set_title('Métricas de Evaluación por Ruta', fontsize=13,
                 fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/07_tabla_metricas.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Guardada: 07_tabla_metricas.png")