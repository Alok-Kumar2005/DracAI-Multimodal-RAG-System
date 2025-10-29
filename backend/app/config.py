from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    """Application settings and configuration."""
    ### processing configurationa
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_results: int = 5
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()