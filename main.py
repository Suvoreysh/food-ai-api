from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Silence TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = FastAPI()

# Load model for inference only
model = tf.keras.models.load_model(
    "food_model.keras",
    compile=False
)

CONFIDENCE_THRESHOLD = 0.3

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = image.resize((224, 224))

    img = np.array(image, dtype=np.float32)
    img = np.expand_dims(img, axis=0)

    # IMPORTANT: correct preprocessing for MobileNetV2
    img = preprocess_input(img)

    preds = model(img, training=False).numpy()

    class_id = int(np.argmax(preds))
    confidence = float(np.max(preds))

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "food": "unknown",
            "confidence": confidence,
            "message": "Image unclear. Please upload a clearer photo."
        }

    return {
        "class_id": class_id,
        "confidence": confidence
    }
