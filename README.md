# Driver Distraction & Travel Recommendation API

API combinada que integra dos sistemas principales:

1. **Driver Distraction Detection** — Detección de comportamientos de conducción utilizando **Deep Learning** y **Transfer Learning** con **ResNet18**.
2. **Travel Recommendation System** — Sistema de recomendación de destinos de viaje basado en **Neural Collaborative Filtering (NCF)** híbrido.

El proyecto permite entrenar modelos de clasificación para identificar comportamientos de conducción distraída y generar recomendaciones personalizadas de destinos, exponiendo ambos mediante una API construida con **FastAPI**.

---

# Dataset

El proyecto utiliza el siguiente dataset de Kaggle:

[Multi-Class Driver Behavior Image Dataset](https://www.kaggle.com/datasets/arafatsahinafridi/multi-class-driver-behavior-image-dataset/data)

El dataset contiene imágenes reales de conductores en diferentes situaciones de manejo:

- `safe_driving`
- `turning`
- `texting_phone`
- `talking_phone`
- `others`

Durante el preprocesamiento, las clases:

- `texting_phone`
- `talking_phone`

son fusionadas en una sola categoría:

- `using_phone`

Por lo tanto, el modelo final trabaja con 4 clases:

- `safe_driving`
- `turning`
- `using_phone`
- `others`

---

# Estructura esperada del dataset

Después de descargar el dataset, se recomienda mantener la siguiente estructura de carpetas:

```text
data_original/
│
├── others/
├── safe_driving/
├── talking_phone/
├── texting_phone/
└── turning/
```

> IMPORTANTE:
>
> El código espera que la carpeta principal del dataset se llame `data_original`.
> Si se utiliza otro nombre, debe modificarse manualmente en el notebook.
t
---

# Tecnologías utilizadas

- Python
- PyTorch
- Torchvision
- FastAPI
- Uvicorn
- Scikit-learn
- Matplotlib
- Seaborn

---

# Entrenamiento del modelo

El notebook realiza las siguientes etapas:

1. División del dataset en entrenamiento y validación.
2. Aumento de datos (*Data Augmentation*).
3. Transfer Learning utilizando ResNet18 preentrenado en ImageNet.
4. Entrenamiento del modelo.
5. Evaluación mediante:
   - Accuracy
   - Classification Report
   - Confusion Matrix
6. Guardado del mejor modelo:

```text
best_driver_behavior_model.pth
```

---

# Instalación

Clonar el repositorio:

```bash
git clone <repo-url>
cd <repo>
```

Crear entorno virtual:

```bash
python -m venv venv
```

Activar entorno virtual:

## Windows

```bash
venv\Scripts\activate
```

## Linux / Mac

```bash
source venv/bin/activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

---

# Ejecución de la API

El proyecto utiliza FastAPI para exponer el modelo entrenado como un servicio REST.

Ejecutar localmente:

```bash
uvicorn main:app --reload
```

La API estará disponible en:

```text
http://127.0.0.1:8000
```

---

# Endpoints

## Home

```http
GET /
```

### Respuesta

```json
{
  "message": "Driver Distraction API Running",
  "classes": [
    "others",
    "safe_driving",
    "turning",
    "using_phone"
  ]
}
```

---

## Predicción de imágenes

```http
POST /predict
```

Permite enviar una imagen y obtener la predicción del comportamiento del conductor.

### Ejemplo usando cURL

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "accept: application/json" \
-H "Content-Type: multipart/form-data" \
-F "file=@imagen.jpg"
```

---

# Respuesta esperada

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

---

# Niveles de peligro

El sistema asigna un nivel de riesgo según el comportamiento detectado:

| Clase | Nivel |
|---|---|
| safe_driving | LOW |
| turning | MEDIUM |
| using_phone | HIGH |
| others | HIGH |

---

---

# Endpoints de Recomendación de Viajes

La API también incluye un completo sistema de recomendación de destinos de viaje basado en **Neural Collaborative Filtering (NCF)** híbrido.

## Health Check

```http
GET /health
```

Verifica que la API y los modelos estén operativos.

### Respuesta

```json
{
  "status": "ok",
  "model_ready": true,
  "device": "cpu"
}
```

---

## Catálogos

### Obtener Usuarios (Paginado)

```http
GET /users?page=1&limit=20&search=
```

**Parámetros:**
- `page` — Número de página (por defecto: 1)
- `limit` — Registros por página, máx 100 (por defecto: 20)
- `search` — Buscar por nombre o email (opcional)

### Respuesta

```json
{
  "users": [
    {
      "user_id": 1,
      "name": "Kavya",
      "email": "kavya@example.com",
      "preferences": "Beaches, Historical",
      "gender": "Female",
      "n_adults": 1,
      "n_children": 0
    }
  ],
  "total": 999,
  "page": 1,
  "limit": 20
}
```

---

### Obtener Destinos (Paginado)

```http
GET /destinations?page=1&limit=20&type=
```

**Parámetros:**
- `page` — Número de página (por defecto: 1)
- `limit` — Registros por página, máx 100 (por defecto: 20)
- `type` — Filtrar por tipo de destino: `Beach`, `Historical`, `Nature`, `Adventure`, `City` (opcional)

### Respuesta

```json
{
  "destinations": [
    {
      "destination_id": 1,
      "name": "Taj Mahal",
      "state": "Uttar Pradesh",
      "type": "Historical",
      "popularity": 8.69,
      "best_time_to_visit": "Nov-Feb"
    }
  ],
  "total": 1000,
  "page": 1,
  "limit": 20
}
```

---

### Obtener Tipos de Destino

```http
GET /destination-types
```

Devuelve los tipos de destino disponibles en el sistema.

### Respuesta

```json
{
  "types": ["Adventure", "Beach", "City", "Historical", "Nature"]
}
```

---

### Obtener Opciones de Preferencia

```http
GET /preference-options
```

Devuelve las preferencias válidas para el formulario de nuevo usuario.

### Respuesta

```json
{
  "preferences": ["Beaches", "Historical", "Nature", "Adventure", "City"]
}
```

---

## Recomendaciones

### Recomendaciones para Usuario Existente

```http
GET /recommend/{user_id}?top_k=5&travel_month=
```

Genera recomendaciones personalizadas para un usuario existente. Utiliza el modelo NCF híbrido (embeddings colaborativos + preferencias de contenido). Los destinos ya visitados se excluyen automáticamente.

**Parámetros:**
- `user_id` — ID del usuario (requerido, path parameter)
- `top_k` — Número de recomendaciones (1-20, por defecto: 5)
- `travel_month` — Mes de viaje (1-12) para aplicar filtro de temporada (opcional)

### Respuesta

```json
{
  "user_id": 1,
  "user_name": "Kavya",
  "email": "kavya@example.com",
  "preferences": "Beaches, Historical",
  "recommendations": [
    {
      "destination_id": 472,
      "name": "Goa Beaches",
      "state": "Goa",
      "type": "Beach",
      "popularity": 9.28,
      "best_time_to_visit": "Nov-Mar",
      "score": 0.9832,
      "match_reason": "Coincide con tus preferencias",
      "season_match": null
    }
  ]
}
```

---

### Recomendaciones para Usuario Nuevo (Cold-Start)

```http
POST /recommend/new-user
```

Genera recomendaciones para un usuario nuevo sin historial previo. Utiliza preferencias declaradas y datos demográficos.

**Body (JSON):**

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

**Parámetros:**
- `preferences` — Preferencias separadas por coma (requerido). Valores válidos: `Beaches`, `Historical`, `Nature`, `Adventure`, `City`
- `gender` — Género (`Male` o `Female`, por defecto: `Male`)
- `n_adults` — Número de adultos en el grupo (1-20, por defecto: 1)
- `n_children` — Número de niños en el grupo (0-20, por defecto: 0)
- `top_k` — Número de recomendaciones (1-20, por defecto: 5)
- `travel_month` — Mes de viaje (1-12) para filtro de temporada (opcional)

### Respuesta

```json
{
  "user_id": null,
  "user_name": "Nuevo usuario",
  "email": null,
  "preferences": "Beaches, Adventure",
  "recommendations": [
    {
      "destination_id": 472,
      "name": "Goa Beaches",
      "state": "Goa",
      "type": "Beach",
      "popularity": 9.28,
      "best_time_to_visit": "Nov-Mar",
      "score": 0.9608,
      "match_reason": "Coincide con tus preferencias",
      "season_match": null
    }
  ]
}
```

---

### Obtener Usuarios Target para un Nuevo Destino

```http
GET /destinations/{destination_type}/target-users?top_k=10
```

Dado el tipo de un destino nuevo, devuelve los usuarios con mayor afinidad para campañas de marketing dirigido.

**Parámetros:**
- `destination_type` — Tipo de destino (path parameter): `Beach`, `Historical`, `Nature`, `Adventure`, `City`
- `top_k` — Número de usuarios a devolver (1-100, por defecto: 10)

### Respuesta

```json
{
  "destination_type": "Beach",
  "target_users": [
    {
      "user_id": 1,
      "name": "Kavya",
      "email": "kavya@example.com",
      "preferences": "Beaches, Historical",
      "group_size": 1
    }
  ]
}
```

---

# Estructura de Archivos

```text
driver-distraction-api/
│
├── artifacts/                           # Modelos entrenados
│   ├── best_driver_behavior_model.pth   # Modelo ResNet18 para distracción
│   ├── travel_recomendations.pth        # Modelo NCF para recomendaciones
│   └── travel_recomendations_state.joblib
│
├── main.py                              # API FastAPI principal
├── recommender.py                       # Motor de recomendaciones
├── requirements.txt                     # Dependencias
├── modelo.ipynb                         # Notebook de entrenamiento (distracción)
└── README.md                            # Este archivo
```

---

# Consideraciones

- El modelo de distracción fue entrenado utilizando imágenes redimensionadas a `224x224`.
- Las imágenes deben corresponder a escenarios similares al dataset original para obtener mejores resultados.
- El modelo de distracción utiliza normalización estándar de ImageNet debido al uso de Transfer Learning con ResNet18.
- El sistema de recomendaciones utiliza un modelo NCF híbrido que combina:
  - **Collaborative Filtering** — Baseado en patrones de comportamiento de usuarios similares
  - **Content-Based Filtering** — Baseado en preferencias declaradas y características del destino
- El filtro de mes de viaje (`travel_month`) es un filtro de contenido adicional que no es parte del modelo NCF.