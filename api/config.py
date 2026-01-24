"""
API Configuration
=================
Environment-based configuration for the FastAPI backend.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]
    
    # WebSocket
    ws_heartbeat_interval: float = 5.0
    ws_data_interval: float = 0.25  # 4 Hz
    
    # Simulation
    default_patient_count: int = 4
    max_patient_count: int = 20
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_prefix = "SENTINEL_"
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
