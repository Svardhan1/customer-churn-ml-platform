import logging

import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from pydantic import BaseModel, StrictBool
from typing import Literal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

MODEL_URI = "models:/CustomerChurnModel/1"
DECISION_THRESHOLD = 0.32


class CustomerInput(BaseModel):
    customer_id: str

    gender: Literal["Male", "Female"]
    senior_citizen: StrictBool
    partner: StrictBool
    dependents: StrictBool
    tenure_months: int

    phone_service: StrictBool
    multiple_lines: Literal["Yes", "No", "No phone service"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    online_security: Literal["Yes", "No", "No internet service"]
    online_backup: Literal["Yes", "No", "No internet service"]
    device_protection: Literal["Yes", "No", "No internet service"]
    tech_support: Literal["Yes", "No", "No internet service"]
    streaming_tv: Literal["Yes", "No", "No internet service"]
    streaming_movies: Literal["Yes", "No", "No internet service"]

    contract_type: Literal["Month-to-month", "One year", "Two year"]
    paperless_billing: StrictBool
    payment_method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]

    monthly_charges: float
    total_charges: float

class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    predicted_churn: bool
    threshold: float
    model: str
    model_version: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    try:
        logger.info("Loading registered model...")
        model = mlflow.sklearn.load_model(MODEL_URI)
        logger.info("Registered model loaded successfully.")

    except Exception:
        logger.exception("Failed to load registered model.")
        model = None

    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production-style API for customer churn prediction.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available.",
        )

    return {
        "status": "healthy",
        "model": "CustomerChurnModel",
        "model_version": "1",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerInput):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available.",
        )

    try:
        customer_data = customer.model_dump()
        customer_id = customer_data.pop("customer_id")

        input_df = pd.DataFrame([customer_data])

        probability = model.predict_proba(input_df)[:, 1][0]

        prediction = int(
            probability >= DECISION_THRESHOLD
        )

        logger.info(
            "Prediction generated | customer_id=%s | probability=%.4f | predicted_churn=%s",
            customer_id,
            probability,
            bool(prediction),
        )

        return {
            "customer_id": customer_id,
            "churn_probability": round(float(probability), 4),
            "predicted_churn": bool(prediction),
            "threshold": DECISION_THRESHOLD,
            "model": "CustomerChurnModel",
            "model_version": "1",
        }

    except Exception:
        logger.exception(
            "Prediction failed for customer_id=%s",
            customer.customer_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        )