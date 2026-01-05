from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # Added
import numpy as np
import tensorflow as tf
import joblib
from pydantic import BaseModel

app = FastAPI(title="Baby Growth Prediction API")

# --- ADD THIS BLOCK TO FIX CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML artifacts
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

        if raw_data.shape != (7, 9):
            return {"error": f"Input must be shape (7, 9), got {raw_data.shape}"}

        scaled_data = scaler.transform(raw_data).reshape(1, 7, 9)
        prediction = model.predict(scaled_data)

        # Updated keys to match what your React Native code expects
        return {
            "w_p50": float(prediction[1][0][0]),
            "h_p50": float(prediction[4][0][0])
        }

    except Exception as e:
        return {"error": str(e)}
