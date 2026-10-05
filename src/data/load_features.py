import os
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine


# Load environment variables
load_dotenv()


# Database configuration
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")


# Create database connection URL
DATABASE_URL = (
    f"postgresql+psycopg2://"
    f"{quote_plus(DB_USER)}:{quote_plus(DB_PASSWORD)}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


engine = create_engine(DATABASE_URL)


def load_features():
    """Load the ML feature dataset from PostgreSQL."""

    project_root = Path(__file__).resolve().parents[2]

    sql_path = project_root / "sql" / "features.sql"

    with open(sql_path, "r", encoding="utf-8") as file:
        query = file.read()

    df = pd.read_sql(query, engine)

    print("Feature data loaded successfully.")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    return df


if __name__ == "__main__":
    df = load_features()

    print("\n--- FEATURE DATA ---")
    print(df.head())

    print("\n--- DATA TYPES ---")
    print(df.dtypes)