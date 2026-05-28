from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
import torch
from torchvision import transforms, models
import io

from recommender import TravelRecommender

# ══════════════════════════════════════════════════════════════════════════════
# Instancia global del motor de recomendación
# ══════════════════════════════════════════════════════════════════════════════
engine = TravelRecommender()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo de recomendación al arrancar el servidor."""
    engine.startup()
    yield


app = FastAPI(
    title="Driver Distraction & Travel Recommendation API",
    description="API combinada con predicción de distracción del conductor y recomendación de destinos de viaje.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://smart-transport-ai-front.vercel.app", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════════════════════
# Pydantic Models for Travel Recommendations
# ══════════════════════════════════════════════════════════════════════════════

class NewUserRequest(BaseModel):
    preferences: str = Field(
        ...,
        examples=["Beaches, Adventure"],
        description="Preferencias separadas por coma. "
                    "Valores válidos: Beaches, Historical, Nature, Adventure, City",
    )
    gender: str = Field("Male", examples=["Male", "Female"])
    n_adults: int = Field(1, ge=1, le=20, description="Número de adultos en el grupo")
    n_children: int = Field(0, ge=0, le=20, description="Número de niños en el grupo")
    top_k: int = Field(5, ge=1, le=20, description="Número de recomendaciones")
    travel_month: Optional[int] = Field(
        None, ge=1, le=12, description="Mes de viaje (1-12) para filtro de temporada"
    )


class RecommendationItem(BaseModel):
    destination_id: int
    name: str
    state: str
    type: str
    popularity: float
    best_time_to_visit: str
    score: float
    match_reason: str
    season_match: Optional[bool] = None


class RecommendationResponse(BaseModel):
    user_id: Optional[int]
    user_name: str
    email: Optional[str] = None
    preferences: str
    recommendations: list[RecommendationItem]


class UserItem(BaseModel):
    user_id: int
    name: str
    email: str
    preferences: str
    gender: str
    n_adults: int
    n_children: int


class DestinationItem(BaseModel):
    destination_id: int
    name: str
    state: str
    type: str
    popularity: float
    best_time_to_visit: str


class TargetUserItem(BaseModel):
    user_id: int
    name: str
    email: str
    preferences: str
    group_size: int


# ══════════════════════════════════════════════════════════════════════════════
# Driver Distraction Configuration
# ══════════════════════════════════════════════════════════════════════════════

# Niveles de peligro
danger_levels = {
    "others": "HIGH",
    "safe_driving": "LOW",
    "using_phone": "HIGH",
    "turning": "MEDIUM"
}

# Device
device = torch.device("cpu")

# Cargar checkpoint
import os
model_path = os.path.join(os.path.dirname(__file__), "artifacts", "best_driver_behavior_model.pth")
checkpoint = torch.load(
    model_path,
    map_location=device
)

# Obtener clases automáticamente
classes = checkpoint["classes"]

# Crear modelo
model = models.resnet18(weights=None)

num_features = model.fc.in_features

model.fc = torch.nn.Linear(
    num_features,
    len(classes)
)

# Cargar pesos
model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

# Transformaciones
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

@app.get("/")
def home():

    return {
        "message": "Driver Distraction API Running",
        "classes": classes
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    image = transform(image).unsqueeze(0)

    with torch.no_grad():

        outputs = model(image)

        probs = torch.softmax(outputs, dim=1)

        confidence, pred = torch.max(probs, 1)

    predicted_class = classes[pred.item()]

    return {
        "class": predicted_class,
        "danger_level": danger_levels[predicted_class],
        "confidence": round(confidence.item(), 4),
        "probabilities": {
            classes[i]: round(probs[0][i].item(), 4)
            for i in range(len(classes))
        }
    }


# ══════════════════════════════════════════════════════════════════════════════
# Travel Recommendation Endpoints
# ══════════════════════════════════════════════════════════════════════════════

def _check_ready():
    """Guard: verifica que el motor de recomendación esté listo."""
    if not engine.ready:
        raise HTTPException(
            status_code=503,
            detail="El modelo aún no ha terminado de inicializarse. Intenta en unos segundos.",
        )


@app.get("/health", tags=["Sistema"])
def health():
    """
    Verifica que la API y el modelo estén operativos.

    **Respuesta:**
    ```json
    { "status": "ok", "model_ready": true, "device": "cpu" }
    ```
    """
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    return {"status": "ok", "model_ready": engine.ready, "device": device}


@app.get("/users", tags=["Catálogos"], response_model=dict)
def get_users(
    page:   int = Query(1, ge=1,   description="Página"),
    limit:  int = Query(20, ge=1, le=100, description="Registros por página"),
    search: str = Query("",        description="Buscar por nombre o email"),
):
    """
    Lista paginada de usuarios.

    **Parámetros:**
    - `page`: número de página (por defecto 1)
    - `limit`: resultados por página (máx 100)
    - `search`: filtrar por nombre o email

    **Respuesta:**
    ```json
    {
      "users": [ { "user_id": 1, "name": "Kavya", "email": "…", … } ],
      "total": 999,
      "page": 1,
      "limit": 20
    }
    ```
    """
    _check_ready()
    return engine.list_users(page=page, limit=limit, search=search)


@app.get("/destinations", tags=["Catálogos"], response_model=dict)
def get_destinations(
    page:  int = Query(1,  ge=1, description="Página"),
    limit: int = Query(20, ge=1, le=100, description="Registros por página"),
    type:  str = Query("", description="Filtrar por tipo: Beach, Historical, Nature, Adventure, City"),
):
    """
    Lista paginada de destinos disponibles.

    **Respuesta:**
    ```json
    {
      "destinations": [ { "destination_id": 1, "name": "Taj Mahal", … } ],
      "total": 1000,
      "page": 1,
      "limit": 20
    }
    ```
    """
    _check_ready()
    return engine.list_destinations(page=page, limit=limit, dest_type=type)


@app.get("/destination-types", tags=["Catálogos"])
def get_destination_types():
    """Devuelve los tipos de destino disponibles."""
    _check_ready()
    return {"types": engine.destination_types()}


@app.get("/preference-options", tags=["Catálogos"])
def get_preference_options():
    """Devuelve las preferencias válidas para el formulario de nuevo usuario."""
    _check_ready()
    return {"preferences": engine.preference_options()}


@app.get(
    "/recommend/{user_id}",
    tags=["Recomendaciones"],
    response_model=RecommendationResponse,
)
def recommend_existing_user(
    user_id: int,
    top_k: int = Query(5, ge=1, le=20, description="Número de recomendaciones"),
    travel_month: Optional[int] = Query(
        None, ge=1, le=12, description="Mes de viaje (1-12) para filtro de temporada"
    ),
):
    """
    Genera recomendaciones personalizadas para un **usuario existente**.

    Utiliza el NCF híbrido (embeddings colaborativos + preferencias de contenido).
    Los destinos ya visitados se excluyen automáticamente.

    **Parámetros opcionales:**
    - `travel_month` (1-12): Mes de viaje para aplicar un bonus de temporada.

    **Path param:** `user_id` — ID del usuario en la base de datos.

    **Respuesta:**
    ```json
    {
      "user_id": 2,
      "user_name": "Rohan",
      "preferences": "Nature, Adventure",
      "recommendations": [ … ]
    }
    ```
    """
    _check_ready()
    try:
        return engine.recommend_for_user(user_id=user_id, top_k=top_k,
                                         mes_viaje=travel_month)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post(
    "/recommend/new-user",
    tags=["Recomendaciones"],
    response_model=RecommendationResponse,
)
def recommend_new_user(body: NewUserRequest):
    """
    Genera recomendaciones para un **usuario nuevo** (cold-start).

    No requiere historial previo. El sistema utiliza las preferencias declaradas
    y datos demográficos para construir un vector de features y recomendar
    destinos mediante el componente de contenido + NCF con peso reducido.

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
    """
    _check_ready()
    return engine.recommend_new_user(
        preferences=body.preferences,
        gender=body.gender,
        n_adults=body.n_adults,
        n_children=body.n_children,
        top_k=body.top_k,
        mes_viaje=body.travel_month,
    )


@app.get(
    "/destinations/{destination_type}/target-users",
    tags=["Recomendaciones"],
    response_model=dict,
)
def target_users_for_destination(
    destination_type: str,
    top_k: int = Query(10, ge=1, le=100, description="Número de usuarios a devolver"),
):
    """
    Dado el **tipo** de un destino nuevo, devuelve los usuarios con mayor
    afinidad para campañas de marketing dirigido.

    **Path param:** `destination_type` — uno de: Beach, Historical, Nature, Adventure, City

    **Respuesta:**
    ```json
    {
      "destination_type": "Nature",
      "target_users": [ { "user_id": 2, "name": "Rohan", … } ]
    }
    ```
    """
    _check_ready()
    valid_types = engine.destination_types()
    if destination_type not in valid_types:
        raise HTTPException(
            status_code=422,
            detail=f"Tipo inválido '{destination_type}'. Válidos: {valid_types}",
        )
    users = engine.users_for_new_destination(destination_type, top_k=top_k)
    return {"destination_type": destination_type, "target_users": users}