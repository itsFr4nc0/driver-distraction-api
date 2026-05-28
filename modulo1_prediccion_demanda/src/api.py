"""
API REST del Módulo 1 — Predicción de Demanda de Transporte
Expone los resultados del modelo LSTM via HTTP para consumo del frontend.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json

app = FastAPI(title="SmartTransport - Módulo 1: Predicción de Demanda")

# Permitir peticiones desde el frontend React (localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Cargar datos pre-generados al iniciar
df_pred = pd.read_csv("reports/predicciones_30_dias.csv", parse_dates=["fecha"])
with open("reports/metricas_evaluacion.json") as f:
    metricas = json.load(f)
df_historico = pd.read_csv("data/processed/demanda_diaria.csv", parse_dates=["travel_date"])


@app.get("/")
def root():
    return {"status": "ok", "modulo": "Predicción de Demanda LSTM"}


@app.get("/rutas")
def get_rutas():
    """Lista de rutas disponibles."""
    return {"rutas": df_pred["ruta"].unique().tolist()}


@app.get("/prediccion/{ruta}")
def get_prediccion(ruta: str):
    """
    Predicción de los próximos 30 días para una ruta.
    Formato compatible con Recharts (array de {date, value}).
    """
    df_ruta = df_pred[df_pred["ruta"] == ruta]
    if df_ruta.empty:
        return {"error": f"Ruta '{ruta}' no encontrada"}

    datos = [
        {"date": str(row["fecha"].date()), "value": int(row["demanda_predicha"])}
        for _, row in df_ruta.iterrows()
    ]
    return {
        "ruta": ruta,
        "prediccion": datos,
        "metricas": metricas.get(ruta, {})
    }


@app.get("/historico/{ruta}")
def get_historico(ruta: str):
    """
    Últimos 90 días históricos de demanda real para una ruta.
    Formato compatible con Recharts.
    """
    df_ruta = df_historico[df_historico["travel_from"] == ruta].tail(90)
    if df_ruta.empty:
        return {"error": f"Ruta '{ruta}' no encontrada"}

    datos = [
        {"date": str(row["travel_date"].date()), "value": int(row["demanda"])}
        for _, row in df_ruta.iterrows()
    ]
    return {"ruta": ruta, "historico": datos}


@app.get("/prediccion-completa/{ruta}")
def get_prediccion_completa(ruta: str):
    """
    Endpoint combinado: histórico + predicción en un solo llamado.
    Es lo más conveniente para que el frontend grafique todo junto.
    """
    historico = get_historico(ruta).get("historico", [])
    prediccion = get_prediccion(ruta).get("prediccion", [])
    metricas_ruta = metricas.get(ruta, {})

    return {
        "ruta": ruta,
        "historico": historico,
        "prediccion": prediccion,
        "metricas": metricas_ruta
    }