from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from torchvision import transforms, models
import io

app = FastAPI()

# Niveles de peligro
danger_levels = {
    "others": "HIGH",
    "safe_driving": "LOW",
    "talking_phone": "HIGH",
    "texting_phone": "HIGH",
    "turning": "MEDIUM"
}

# Device
device = torch.device("cpu")

# Cargar checkpoint
checkpoint = torch.load(
    "driver_behavior_model.pth",
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
    transforms.Resize((192, 192)),
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