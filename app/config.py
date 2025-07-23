## Step 3: Configuration (app/config.py)

"""Configuration settings for the evaluation backend."""
import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # App settings
    app_name: str = "Claude Evaluation Backend"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    # MongoDB settings
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_database: str = "evaluation_db"
    
    # OpenTelemetry settings
    otel_service_name: str = "evaluation-service"
    otel_service_version: str = "1.0.0"
    jaeger_endpoint: str = "http://localhost:14268/api/traces"
    prometheus_port: int = 8001
    
    # Logging settings
    log_level: str = "INFO"
    log_file_path: str = "logs/app.log"
    
    class Config:
        env_file = ".env"


settings = Settings()

