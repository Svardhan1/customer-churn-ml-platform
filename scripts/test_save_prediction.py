
from sqlalchemy import text

from src.db.connection import engine
from src.db.repository import register_model_version, save_prediction

# Reuse the registered model version.
model_version_id = register_model_version(
    model_name="CustomerChurnModel",
    version="1",
    algorithm="XGBoost",
    accuracy=0.7587,
    precision_score=0.5331,
    recall_score=0.7326,
    f1_score=0.6171,
    roc_auc=0.8417,
)

# Use a real customer ID from the local database.
with engine.connect() as connection:
    customer_id = connection.execute(
        text("SELECT customer_id FROM customers LIMIT 1")
    ).scalar_one()

# Save a sample prediction.
prediction_id = save_prediction(
    customer_id=customer_id,
    model_version_id=model_version_id,
    churn_probability=0.7121,
    predicted_churn=True,
)

print("Prediction saved successfully.")
print("Customer ID:", customer_id)
print("Model version ID:", model_version_id)
print("Prediction ID:", prediction_id)
