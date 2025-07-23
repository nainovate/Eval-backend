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
    
    # OpenTelemetry settings - Updated service name
    otel_service_name: str = "claude-evaluation-service"
    otel_service_version: str = "1.0.0"
    
    # Jaeger endpoint - will be dynamically set based on environment
    jaeger_endpoint: str = "http://localhost:14268/api/traces"
    
    # Prometheus settings
    prometheus_port: int = 8001
    
    # Logging settings
    log_level: str = "INFO"
    log_file_path: str = "logs/app.log"
    
    # Environment detection
    docker_env: bool = False
    
    class Config:
        env_file = ".env"

    def __post_init__(self):
        """Post-initialization to set dynamic values."""
        # Detect if running in Docker and adjust endpoints
        if os.getenv("DOCKER_ENV") == "true" or self.docker_env:
            self.jaeger_endpoint = "http://jaeger:14268/api/traces"
            self.mongodb_url = "mongodb://mongodb:27017"
        
        # Allow override via environment variable
        if os.getenv("JAEGER_ENDPOINT"):
            self.jaeger_endpoint = os.getenv("JAEGER_ENDPOINT")


settings = Settings()

# Call post-init manually since Pydantic doesn't call it automatically
if hasattr(settings, '__post_init__'):
    settings.__post_init__()