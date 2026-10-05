import os
from dotenv import load_dotenv

load_dotenv()

print("DB HOST:", os.getenv("DB_HOST"))
print("DB PORT:", os.getenv("DB_PORT"))
print("DB NAME:", os.getenv("DB_NAME"))
print("DB USER:", os.getenv("DB_USER"))

print("Password loaded:", bool(os.getenv("DB_PASSWORD")))