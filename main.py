from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# Silence TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = FastAPI()

# IMPORTANT: compile=False
model = tf.keras.models.load_model(
    "food_model.keras",
    compile=False
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = image.resize((224, 224))

    img = np.array(image, dtype=np.float32)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model(img, training=False)
    preds = preds.numpy()

    class_id = int(np.argmax(preds))
    confidence = float(np.max(preds))

    return {
        "class_id": class_id,
        "confidence": confidence
    }
