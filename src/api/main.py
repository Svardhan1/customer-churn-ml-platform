import os
from pathlib import Path
import logging
from contextlib import asynccontextmanager
from typing import Literal

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, StrictBool

from src.db.repository import (
    register_model_version,
    save_prediction,
)


# --------------------------------------------------
# Logging
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# Model configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_URI = os.getenv(
    "MODEL_URI",
    str(
        PROJECT_ROOT
        / "mlruns"
        / "1"
        / "models"
        / "m-982ea91903524a629eff12549bd43184"
        / "artifacts"
    ),
)

MODEL_NAME = "CustomerChurnModel"
MODEL_VERSION = "1"
DECISION_THRESHOLD = 0.32

model = None
model_version_id = None


# --------------------------------------------------
# Request and response schemas
# --------------------------------------------------

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
    prediction_id: int
    customer_id: str
    churn_probability: float
    predicted_churn: bool
    threshold: float
    model: str
    model_version: str


# --------------------------------------------------
# Application startup and shutdown
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_version_id

    try:
        logger.info("Loading registered ML model...")

        model = mlflow.sklearn.load_model(MODEL_URI)

        logger.info("ML model loaded successfully.")

        logger.info("Registering model metadata in PostgreSQL...")

        model_version_id = register_model_version(
            model_name=MODEL_NAME,
            version=MODEL_VERSION,
            algorithm="XGBoost",
            accuracy=0.7587,
            precision_score=0.5331,
            recall_score=0.7326,
            f1_score=0.6171,
            roc_auc=0.8417,
        )

        logger.info(
            "Model metadata ready | model_version_id=%s",
            model_version_id,
        )

    except Exception:
        logger.exception(
            "Application startup failed: model loading or database initialization failed."
        )
        model = None
        model_version_id = None

    yield

    logger.info("Application shutdown complete.")


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Production-style customer churn prediction API "
        "with PostgreSQL prediction persistence."
    ),
    version="1.1.0",
    lifespan=lifespan,
)


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
def health_check():
    if model is None or model_version_id is None:
        raise HTTPException(
            status_code=503,
            detail="Model or database initialization is unavailable.",
        )

    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "database": "initialized",
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerInput):
    if model is None or model_version_id is None:
        raise HTTPException(
            status_code=503,
            detail="Model or database initialization is unavailable.",
        )

    customer_id = customer.customer_id

    try:
        customer_data = customer.model_dump()
        customer_data.pop("customer_id")

        input_df = pd.DataFrame([customer_data])

        probability = float(
            model.predict_proba(input_df)[0, 1]
        )

        predicted_churn = probability >= DECISION_THRESHOLD

        logger.info(
            "Prediction generated | customer_id=%s | "
            "probability=%.4f | predicted_churn=%s",
            customer_id,
            probability,
            predicted_churn,
        )

    except Exception as exc:
        logger.exception(
            "Model prediction failed | customer_id=%s",
            customer_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Model prediction failed.",
        ) from exc

    # Persist the prediction separately so database failures
    # can be reported accurately.
    try:
        prediction_id = save_prediction(
            customer_id=customer_id,
            model_version_id=model_version_id,
            churn_probability=probability,
            predicted_churn=predicted_churn,
        )

        logger.info(
            "Prediction persisted | prediction_id=%s | customer_id=%s",
            prediction_id,
            customer_id,
        )

    except Exception as exc:
        logger.exception(
            "Prediction generated but database persistence failed | "
            "customer_id=%s",
            customer_id,
        )

        raise HTTPException(
            status_code=503,
            detail="Prediction generated, but saving it to the database failed.",
        ) from exc

    return PredictionResponse(
        prediction_id=prediction_id,
        customer_id=customer_id,
        churn_probability=round(probability, 4),
        predicted_churn=predicted_churn,
        threshold=DECISION_THRESHOLD,
        model=MODEL_NAME,
        model_version=MODEL_VERSION,
    )
