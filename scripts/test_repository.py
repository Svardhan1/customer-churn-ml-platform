
from src.db.repository import register_model_version

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

print("Model version ID:", model_version_id)
