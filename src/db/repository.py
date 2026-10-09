from sqlalchemy import text

from src.db.connection import engine



def register_model_version(
    model_name: str,
    version: str,
    algorithm: str,
    accuracy: float,
    precision_score: float,
    recall_score: float,
    f1_score: float,
    roc_auc: float,
) -> int:
    """Return an existing model version ID or register a new version."""

    with engine.begin() as connection:
        result = connection.execute(
            text(
                """
                SELECT model_version_id
                FROM model_versions
                WHERE model_name = :model_name
                  AND version = :version
                ORDER BY model_version_id
                LIMIT 1
                """
            ),
            {
                "model_name": model_name,
                "version": version,
            },
        )

        existing_id = result.scalar_one_or_none()

        if existing_id is not None:
            return existing_id

        result = connection.execute(
            text(
                """
                INSERT INTO model_versions (
                    model_name,
                    version,
                    algorithm,
                    accuracy,
                    precision_score,
                    recall_score,
                    f1_score,
                    roc_auc
                )
                VALUES (
                    :model_name,
                    :version,
                    :algorithm,
                    :accuracy,
                    :precision_score,
                    :recall_score,
                    :f1_score,
                    :roc_auc
                )
                RETURNING model_version_id
                """
            ),
            {
                "model_name": model_name,
                "version": version,
                "algorithm": algorithm,
                "accuracy": accuracy,
                "precision_score": precision_score,
                "recall_score": recall_score,
                "f1_score": f1_score,
                "roc_auc": roc_auc,
            },
        )

        return result.scalar_one()


def save_prediction(
    customer_id: str,
    model_version_id: int,
    churn_probability: float,
    predicted_churn: bool,
):
    query = text(
        """
        INSERT INTO predictions (
            customer_id,
            model_version_id,
            churn_probability,
            predicted_churn
        )
        VALUES (
            :customer_id,
            :model_version_id,
            :churn_probability,
            :predicted_churn
        )
        RETURNING prediction_id;
        """
    )

    with engine.begin() as connection:
        result = connection.execute(
            query,
            {
                "customer_id": customer_id,
                "model_version_id": model_version_id,
                "churn_probability": churn_probability,
                "predicted_churn": predicted_churn,
            },
        )

        return result.scalar_one()