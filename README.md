# Driver Distraction Detection API

Sistema de detección de comportamientos de conducción utilizando **Deep Learning** y **Transfer Learning** con **ResNet18**.

El proyecto permite entrenar un modelo de clasificación de imágenes para identificar comportamientos de conducción distraída y exponerlo mediante una API construida con FastAPI.

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

# Consideraciones

- El modelo fue entrenado utilizando imágenes redimensionadas a `224x224`.
- Las imágenes deben corresponder a escenarios similares al dataset original para obtener mejores resultados.
- El modelo utiliza normalización estándar de ImageNet debido al uso de Transfer Learning con ResNet18.