import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from urllib.parse import quote_plus

from src.data.clean import clean_total_charges


DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

DATABASE_URL = (
    f"postgresql+psycopg2://"
    f"{quote_plus(DB_USER)}:{quote_plus(DB_PASSWORD)}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

engine = create_engine(DATABASE_URL)


def extract_data():
    """Load the raw customer churn CSV into a DataFrame."""

    df = pd.read_csv(DATA_PATH)

    print("Data extracted successfully.")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    return df


def transform_data(df):
    """Clean and transform the raw customer data."""

    # Step 1: Clean TotalCharges
    df = clean_total_charges(df)

    # Step 2: Convert Yes/No columns to boolean
    boolean_columns = [
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
        "Churn",
    ]

    for column in boolean_columns:
        df[column] = df[column].map({
            "Yes": True,
            "No": False,
        })

    # Step 3: Convert SeniorCitizen from 0/1 to boolean
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(bool)

    # Step 4: Rename columns to database naming convention
    df = df.rename(
        columns={
            "customerID": "customer_id",
            "SeniorCitizen": "senior_citizen",
            "Partner": "partner",
            "Dependents": "dependents",
            "tenure": "tenure_months",
            "PhoneService": "phone_service",
            "MultipleLines": "multiple_lines",
            "InternetService": "internet_service",
            "OnlineSecurity": "online_security",
            "OnlineBackup": "online_backup",
            "DeviceProtection": "device_protection",
            "TechSupport": "tech_support",
            "StreamingTV": "streaming_tv",
            "StreamingMovies": "streaming_movies",
            "Contract": "contract_type",
            "PaperlessBilling": "paperless_billing",
            "PaymentMethod": "payment_method",
            "MonthlyCharges": "monthly_charges",
            "TotalCharges": "total_charges",
            "Churn": "churn",
        }
    )

    print("Data transformed successfully.")

    return df

def validate_data(df):
    """Validate the transformed customer data."""

    expected_columns = {
        "customer_id",
        "gender",
        "senior_citizen",
        "partner",
        "dependents",
        "tenure_months",
        "phone_service",
        "multiple_lines",
        "internet_service",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
        "contract_type",
        "paperless_billing",
        "payment_method",
        "monthly_charges",
        "total_charges",
        "churn",
    }

    actual_columns = set(df.columns)

    if actual_columns != expected_columns:
        missing = expected_columns - actual_columns
        extra = actual_columns - expected_columns

        raise ValueError(
            f"Column validation failed. "
            f"Missing: {missing}, Extra: {extra}"
        )

    if len(df) != 7043:
        raise ValueError(
            f"Unexpected row count: {len(df)}"
        )

    if df["customer_id"].isna().any():
        raise ValueError(
            "customer_id contains missing values."
        )

    if df["customer_id"].duplicated().any():
        raise ValueError(
            "Duplicate customer_id values found."
        )

    numeric_columns = [
        "tenure_months",
        "monthly_charges",
        "total_charges",
    ]

    for column in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(
                f"{column} is not numeric."
            )

    boolean_columns = [
        "senior_citizen",
        "partner",
        "dependents",
        "phone_service",
        "paperless_billing",
        "churn",
    ]

    for column in boolean_columns:
        if not pd.api.types.is_bool_dtype(df[column]):
            raise ValueError(
                f"{column} is not boolean."
            )

    print("Data validation successful.")

    return df

def load_data(df):
    """Load validated data into PostgreSQL."""

    customers_df = df[
        [
            "customer_id",
            "gender",
            "senior_citizen",
            "partner",
            "dependents",
            "tenure_months",
            "churn",
        ]
    ].copy()

    services_df = df[
        [
            "customer_id",
            "phone_service",
            "multiple_lines",
            "internet_service",
            "online_security",
            "online_backup",
            "device_protection",
            "tech_support",
            "streaming_tv",
            "streaming_movies",
        ]
    ].copy()

    billing_df = df[
        [
            "customer_id",
            "contract_type",
            "paperless_billing",
            "payment_method",
            "monthly_charges",
            "total_charges",
        ]
    ].copy()

    customers_df.to_sql(
        "customers",
        engine,
        if_exists="append",
        index=False,
    )

    services_df.to_sql(
        "customer_services",
        engine,
        if_exists="append",
        index=False,
    )

    billing_df.to_sql(
        "customer_billing",
        engine,
        if_exists="append",
        index=False,
    )

    print("Data loaded successfully.")


if __name__ == "__main__":
    df = extract_data()
    df = transform_data(df)
    df = validate_data(df)
    load_data(df)
    