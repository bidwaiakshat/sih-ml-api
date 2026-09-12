from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).parent

# Load the scaler and XGBoost model
scaler = joblib.load(BASE_DIR / "price_scaler.pkl")

model = xgb.XGBRegressor()
model.load_model(BASE_DIR / "price_xgboost_model.json")

app = FastAPI(
    title="SIH Cooperative Gig Price API",
    version="1.0"
)

# Allow the Lovable frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PriceRequest(BaseModel):
    difficulty: int = Field(..., ge=1, le=10)
    urgency: int = Field(..., ge=1, le=5)
    is_commercial: int = Field(0, ge=0, le=1)
    hour_of_day: int | None = Field(None, ge=0, le=23)
    band_pct: float = Field(0.15, ge=0.0, le=1.0)


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "ML Price API is running"
    }


@app.post("/predict-price")
def predict_price(req: PriceRequest):

    # Default to current hour if hour isn't supplied
    from datetime import datetime

    hour = (
        req.hour_of_day
        if req.hour_of_day is not None
        else datetime.now().hour
    )

    # IMPORTANT:
    # The scaler was trained on these four features
    features = pd.DataFrame([{
        "Difficulty": req.difficulty,
        "Urgency": req.urgency,
        "Hour_of_Day": hour,
        "Is_Commercial": req.is_commercial,
    }])

    # Apply the same scaling used during training
    scaled_features = scaler.transform(features)

    # Predict price
    base_price = float(model.predict(scaled_features)[0])

    # Create bidding range
    band = base_price * req.band_pct

    min_price = max(base_price - band, 0)
    max_price = base_price + band

    return {
        "base_price": round(base_price, 2),
        "min_price": round(min_price, 2),
        "max_price": round(max_price, 2),
        "inputs_used": {
            "Difficulty": req.difficulty,
            "Urgency": req.urgency,
            "Hour_of_Day": hour,
            "Is_Commercial": req.is_commercial
        }
    }