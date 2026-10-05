import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def prepare_features(df: pd.DataFrame):
    """
    Prepare feature matrix X and target y.

    customer_id is excluded because it is an identifier,
    not a predictive feature.
    """

    df = df.copy()

    y = df["churn"]

    X = df.drop(
        columns=["churn", "customer_id"]
    )

    numerical_features = [
        "tenure_months",
        "monthly_charges",
        "total_charges",
    ]

    boolean_features = [
        "senior_citizen",
        "partner",
        "dependents",
        "phone_service",
        "paperless_billing",
    ]

    categorical_features = [
        "gender",
        "multiple_lines",
        "internet_service",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
        "contract_type",
        "payment_method",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                StandardScaler(),
                numerical_features,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features,
            ),
            (
                "boolean",
                "passthrough",
                boolean_features,
            ),
        ]
    )

    return X, y, preprocessor