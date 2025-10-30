from pydantic_settings import BaseSettings
from pathlib import Path
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings and configuration."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_results: int = 5

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    clip_model: str = "openai/clip-vit-base-patch32"
    chroma_persist_directory: str = "./chroma_db"
    collection_name: str = "multimodal_rag"
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    upload_directory: str = "./data/uploads"
    max_upload_size: int = 10485760  ### 10 MB
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True
    
    def create_directories(self):
        """Create necessary directories if they don't exist."""
        Path(self.upload_directory).mkdir(parents=True, exist_ok=True)
        Path(self.chroma_persist_directory).mkdir(parents=True, exist_ok=True)
        Path(f"{self.upload_directory}/documents").mkdir(parents=True, exist_ok=True)
        Path(f"{self.upload_directory}/images").mkdir(parents=True, exist_ok=True)
        Path(f"{self.upload_directory}/pdfs").mkdir(parents=True, exist_ok=True)


settings = Settings()