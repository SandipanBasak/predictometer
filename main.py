from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# ----------------------------------------------------
# 1️⃣ Create FastAPI app
# ----------------------------------------------------
app = FastAPI(title="Soil Classification API", version="1.0")

# Allow React frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# 2️⃣ Load the trained model
# ----------------------------------------------------
MODEL_PATH = "soil_resnet50_finetuned.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# ----------------------------------------------------
# 3️⃣ Define soil class names (order must match training)
# ----------------------------------------------------
CLASS_NAMES = [
    "alluvial",
    "black",
    "cinder",
    "clay",
    "laterite",
    "loamy",
    "peat",
    "red",
    "sandy",
    "sandy_loam",
    "yellow"
]

# ----------------------------------------------------
# 4️⃣ Static mapping: soil → suitable crops
# ----------------------------------------------------
SOIL_TO_CROPS = {
    "alluvial": [
        "Rice", "Wheat", "Sugarcane", "Maize", "Jute", "Pulses", "Oilseeds"
    ],
    "black": [
        "Cotton", "Soybean", "Sunflower", "Sorghum", "Wheat", "Citrus", "Groundnut"
    ],
    "cinder": [
        "Tapioca", "Cashew", "Coconut", "Arecanut", "Pineapple"
    ],
    "clay": [
        "Rice", "Sugarcane", "Jute", "Paddy", "Vegetables"
    ],
    "laterite": [
        "Tea", "Coffee", "Cashew", "Rubber", "Coconut"
    ],
    "loamy": [
        "Sugarcane", "Cotton", "Wheat", "Pulses", "Oilseeds", "Potato", "Vegetables"
    ],
    "peat": [
        "Rice", "Jute", "Sugarcane", "Vegetables"
    ],
    "red": [
        "Groundnut", "Millets", "Potato", "Rice", "Wheat", "Pulses"
    ],
    "sandy": [
        "Peanuts", "Watermelon", "Potatoes", "Carrots", "Cabbage"
    ],
    "sandy_loam": [
        "Groundnut", "Potato", "Maize", "Tomato", "Onion", "Melons"
    ],
    "yellow": [
        "Maize", "Pulses", "Peas", "Groundnut", "Fruits", "Oilseeds"
    ]
}

# ----------------------------------------------------
# 5️⃣ Health check route
# ----------------------------------------------------
@app.get("/")
def home():
    return {"message": "✅ Soil Classification API is running!"}

# ----------------------------------------------------
# 6️⃣ Prediction route
# ----------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and preprocess the uploaded image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((224, 224))  # ResNet50 input size

        # Convert to array and normalize
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        preds = model.predict(img_array)
        pred_class = CLASS_NAMES[np.argmax(preds[0])]
        confidence = float(np.max(preds[0]))

        # Get recommended crops
        crops = SOIL_TO_CROPS.get(pred_class, ["No data available"])

        # Return response
        return JSONResponse({
            "predicted_class": pred_class,
            "confidence": round(confidence, 3),
            "recommended_crops": crops
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ----------------------------------------------------
# 7️⃣ Run locally
# ----------------------------------------------------
# Run with: uvicorn main:app --reload
