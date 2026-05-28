# API Integrada

API combinada que integra tres sistemas principales:

1. **Driver Distraction Detection** — Detección de comportamientos de conducción utilizando **Deep Learning** y **Transfer Learning** con **ResNet18**.
2. **Travel Recommendation System** — Sistema de recomendación de destinos de viaje basado en **Neural Collaborative Filtering (NCF)** híbrido.
3. **Transport Demand Prediction** — Predicción de demanda de pasajeros por ruta para los próximos 30 días usando redes **LSTM**.

El proyecto permite entrenar modelos de clasificación para identificar comportamientos de conducción distraída, generar recomendaciones personalizadas de destinos, y predecir la demanda de transporte, exponiendo los tres módulos mediante APIs construidas con **FastAPI**.

---

# Módulo 1: Predicción de Demanda de Transporte (Series de Tiempo)

## Dataset

Se utilizó el **Nairobi Transport Demand Dataset** (Kaggle). Contiene 51,645 registros individuales de pasajeros de una empresa de transporte terrestre en Kenia, cubriendo el período octubre 2017 — abril 2018 en 17 rutas de origen.

Las rutas fueron agrupadas en 4 series temporales para el modelado:

| Ruta | Descripción |
|---|---|
| Kisii | Ruta de mayor volumen histórico, con quiebre estructural en marzo 2018 |
| Migori | Serie más estable, mejor desempeño del modelo |
| Homa Bay | Serie con eventos atípicos (Semana Santa) |
| Otras Rutas | Agregado de las 14 rutas restantes |

## Tecnologías utilizadas

- Python 3.12
- TensorFlow / Keras
- Scikit-learn
- Pandas / NumPy
- Matplotlib / Statsmodels
- FastAPI / Uvicorn

## Estructura del módulo

```text
modulo1_prediccion_demanda/
│
├── main.py                         # Punto de entrada del módulo
├── requirements.txt                # Dependencias del módulo
│
├── data/
│   ├── raw/
│   │   └── train_revised.csv       # Dataset original de Nairobi
│   └── processed/
│       ├── demanda_diaria.csv      # Demanda agregada por día y ruta
│       ├── demanda_continua.csv
│       └── demanda_semanal.csv
│
├── models/                         # Modelos LSTM entrenados (.keras)
│
├── notebooks/
│   ├── 01_eda.ipynb                # Análisis exploratorio de datos
│   ├── 03_modelo_regresion.ipynb   # Modelo de regresión base
│   └── 05_modelo_lstm.ipynb        # Modelo LSTM principal
│
├── reports/
│   ├── predicciones_30_dias.csv    # Predicciones futuras por ruta
│   ├── metricas_evaluacion.json    # Métricas finales del modelo
│   └── figures/                   # Gráficas generadas
│       ├── 01_series_historicas.png
│       ├── 02_pred_vs_real_*.png
│       ├── 03_prediccion_futura_*.png
│       ├── 05_descomposicion_*.png
│       ├── 06_demanda_dia_semana.png
│       └── 07_tabla_metricas.png
│
└── src/
    ├── preprocessing.py            # Carga, limpieza y preparación de datos
    ├── model.py                    # Arquitectura del modelo LSTM
    ├── train.py                    # Entrenamiento y evaluación
    ├── visualization.py            # Generación de gráficas
    ├── generar_graficas.py         # Script para regenerar todas las figuras
    └── api.py                      # API REST del módulo
```

## Arquitectura del modelo LSTM

```text
Entrada: (lookback, 1)
→ LSTM(64 unidades, return_sequences=True)
→ Dropout(0.2)
→ LSTM(32 unidades, return_sequences=False)
→ Dropout(0.2)
→ Dense(16, activación ReLU)
→ Dense(1) — demanda normalizada
```

Total de parámetros: 29,857

## Métricas de evaluación

| Ruta | RMSE | MAE | MAPE (%) |
|---|---|---|---|
| Migori | 9.83 | 8.11 | 17.58 |
| Kisii | 11.74 | 8.85 | 42.42 |
| Homa Bay | 29.60 | 24.96 | 271.53* |
| Otras Rutas | 64.65 | 56.17 | 24.89 |

> *El MAPE elevado en Homa Bay se debe a días con demanda cero durante Semana Santa (1-4 abril 2018). El RMSE y MAE son las métricas representativas para esta ruta.

## Instalación y ejecución del módulo

Desde la raíz del repositorio:

```bash
cd modulo1_prediccion_demanda
pip install -r requirements.txt
```

Ejecutar el preprocesamiento:

```bash
python src/preprocessing.py
```

Entrenar los modelos:

```bash
python src/train.py
```

Regenerar gráficas:

```bash
python src/generar_graficas.py
```

Ejecutar la API del módulo:

```bash
uvicorn src.api:app --reload --port 8001
```

## Endpoints del módulo 1

### Rutas disponibles

```http
GET /rutas
```

```json
{"rutas": ["Kisii", "Migori", "Homa Bay", "Otras Rutas"]}
```

### Predicción completa por ruta

```http
GET /prediccion-completa/{ruta}
```

Devuelve histórico de los últimos 90 días + predicción de los próximos 30 días + métricas del modelo.

```json
{
  "ruta": "Migori",
  "historico": [
    {"date": "2018-01-21", "value": 45},
    ...
  ],
  "prediccion": [
    {"date": "2018-04-21", "value": 50},
    ...
  ],
  "metricas": {
    "RMSE": 9.83,
    "MAE": 8.11,
    "MAPE": 17.58
  }
}
```

### Predicción solo futura

```http
GET /prediccion/{ruta}
```

### Histórico solo

```http
GET /historico/{ruta}
```

---

# Módulo 2: Driver Distraction Detection

**Autor:** Alejandro Gómez Franco

## Dataset

El proyecto utiliza el siguiente dataset de Kaggle:

[Multi-Class Driver Behavior Image Dataset](https://www.kaggle.com/datasets/arafatsahinafridi/multi-class-driver-behavior-image-dataset/data)

El dataset contiene imágenes reales de conductores en diferentes situaciones de manejo:

- `safe_driving`
- `turning`
- `texting_phone`
- `talking_phone`
- `others`

Durante el preprocesamiento, las clases `texting_phone` y `talking_phone` son fusionadas en `using_phone`. El modelo final trabaja con 4 clases:

- `safe_driving`
- `turning`
- `using_phone`
- `others`

## Estructura esperada del dataset

```text
data_original/
│
├── others/
├── safe_driving/
├── talking_phone/
├── texting_phone/
└── turning/
```

## Tecnologías utilizadas

- Python
- PyTorch
- Torchvision
- FastAPI
- Uvicorn
- Scikit-learn
- Matplotlib / Seaborn

## Entrenamiento del modelo

El notebook realiza las siguientes etapas:

1. División del dataset en entrenamiento y validación.
2. Aumento de datos (*Data Augmentation*).
3. Transfer Learning utilizando ResNet18 preentrenado en ImageNet.
4. Entrenamiento del modelo.
5. Evaluación mediante Accuracy, Classification Report y Confusion Matrix.
6. Guardado del mejor modelo: `artifacts/best_driver_behavior_model.pth`

---

# Módulo 3: Travel Recommendation System

**Autor:** Alejandro Gómez Franco

Sistema de recomendación de destinos de viaje basado en **Neural Collaborative Filtering (NCF)** híbrido que combina:

- **Collaborative Filtering** — Basado en patrones de comportamiento de usuarios similares
- **Content-Based Filtering** — Basado en preferencias declaradas y características del destino

---

# Instalación general

Clonar el repositorio:

```bash
git clone https://github.com/itsFr4nc0/driver-distraction-api.git
cd driver-distraction-api
```

Crear entorno virtual:

```bash
python -m venv venv
```

Activar entorno virtual:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / Mac:**
```bash
source venv/bin/activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

---

# Ejecución de la API principal

```bash
uvicorn main:app --reload
```

La API estará disponible en `http://127.0.0.1:8000`

---

# Endpoints — Módulo 2: Driver Distraction

## Home

```http
GET /
```

```json
{
  "message": "Driver Distraction API Running",
  "classes": ["others", "safe_driving", "turning", "using_phone"]
}
```

## Predicción de imágenes

```http
POST /predict
```

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "accept: application/json" \
-H "Content-Type: multipart/form-data" \
-F "file=@imagen.jpg"
```

```json
{
  "class": "using_phone",
  "danger_level": "HIGH",
  "confidence": 0.9821,
  "probabilities": {
    "others": 0.0123,
    "safe_driving": 0.0011,
    "turning": 0.0045,
    "using_phone": 0.9821
  }
}
```

## Niveles de peligro

| Clase | Nivel |
|---|---|
| safe_driving | LOW |
| turning | MEDIUM |
| using_phone | HIGH |
| others | HIGH |

---

# Endpoints — Módulo 3: Travel Recommendation

## Health Check

```http
GET /health
```

```json
{"status": "ok", "model_ready": true, "device": "cpu"}
```

## Obtener Usuarios

```http
GET /users?page=1&limit=20&search=
```

## Obtener Destinos

```http
GET /destinations?page=1&limit=20&type=
```

## Recomendaciones para Usuario Existente

```http
GET /recommend/{user_id}?top_k=5&travel_month=
```

## Recomendaciones para Usuario Nuevo

```http
POST /recommend/new-user
```

```json
{
  "preferences": "Beaches, Adventure",
  "gender": "Female",
  "n_adults": 2,
  "n_children": 1,
  "top_k": 5,
  "travel_month": 1
}
```

## Usuarios Target para Nuevo Destino

```http
GET /destinations/{destination_type}/target-users?top_k=10
```

---

# Estructura general del repositorio

```text
driver-distraction-api/
│
├── artifacts/                              # Modelos entrenados
│   ├── best_driver_behavior_model.pth      # Módulo 2 — ResNet18
│   ├── travel_recomendations.pth           # Módulo 3 — NCF
│   └── travel_recomendations_state.joblib
│
├── main.py                                 # API FastAPI principal (módulos 2 y 3)
├── recommender.py                          # Motor de recomendaciones
├── requirements.txt                        # Dependencias generales
├── modelo.ipynb                            # Notebook entrenamiento módulo 2
├── README.md
│
└── modulo1_prediccion_demanda/             # Módulo 1 — Predicción de Demanda
    ├── main.py
    ├── requirements.txt
    ├── data/
    ├── models/
    ├── notebooks/
    ├── reports/
    └── src/
```

---

# Consideraciones

- El modelo de distracción fue entrenado con imágenes redimensionadas a `224x224`.
- Las imágenes deben corresponder a escenarios similares al dataset original.
- El modelo de distracción utiliza normalización estándar de ImageNet.
- El sistema de recomendaciones combina Collaborative Filtering y Content-Based Filtering.
- El filtro de mes de viaje (`travel_month`) es un filtro de contenido adicional.
- Los modelos `.keras` del Módulo 1 no están incluidos en el repositorio por tamaño. Para regenerarlos ejecutar `python src/train.py` desde `modulo1_prediccion_demanda/`.