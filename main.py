from fastapi import FastAPI
import numpy as np
import tensorflow as tf
import joblib
from pydantic import BaseModel

app = FastAPI(title="Baby Growth Prediction API")

# Load ML artifacts at startup (important)
model = tf.keras.models.load_model("growth_model.h5")
scaler = joblib.load("scaler.gz")

class GrowthData(BaseModel):
    data: list

@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/predict")
def predict(input_data: GrowthData):
    try:
        raw_data = np.array(input_data.data)

        # Validate shape (important for mobile errors)
        if raw_data.shape != (7, 9):
            return {"error": "Input must be shape (7, 9)"}

        scaled_data = scaler.transform(raw_data).reshape(1, 7, 9)
        prediction = model.predict(scaled_data)

        return {
            "weight_gain": float(prediction[1][0][0]),
            "height_gain": float(prediction[4][0][0])
        }

    except Exception as e:
        return {"error": str(e)}
