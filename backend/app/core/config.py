# app/core/config.py
from pydantic import BaseModel

class Settings(BaseModel):
    PROJECT_NAME: str = "MLOps Platform"
    API_V1_STR: str = "/api/v1"
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "password"
    POSTGRES_DB: str = "mlops"

    class Config:
        env_file = ".env"

settings = Settings()