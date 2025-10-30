from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    """Application settings and configuration."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_results: int = 5

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    clip_model: str = "openai/clip-vit-base-patch32"
    chroma_persist_directory: str = "./chroma_db"
    collection_name: str = "multimodal_rag"
    gemini_api_key = str
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()