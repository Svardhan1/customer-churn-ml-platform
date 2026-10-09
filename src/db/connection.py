import os

from dotenv import load_dotenv
from sqlalchemy import URL, create_engine, text

load_dotenv()


def get_database_url():
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT", "5432")
    database = os.getenv("DB_NAME")
    username = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")

    required = {
        "DB_HOST": host,
        "DB_NAME": database,
        "DB_USER": username,
        "DB_PASSWORD": password,
    }

    missing = [key for key, value in required.items() if not value]

    if missing:
        raise RuntimeError(
            f"Missing required database environment variables: {missing}"
        )

    return URL.create(
        drivername="postgresql+psycopg2",
        username=username,
        password=password,
        host=host,
        port=int(port),
        database=database,
    )


DATABASE_URL = get_database_url()

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)


def test_database_connection():
    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1"))
        return result.scalar() == 1