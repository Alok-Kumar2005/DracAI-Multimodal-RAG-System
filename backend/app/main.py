from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import sys
import uvicorn

from .config import settings
from .api.routes import router

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/app.log",
    rotation="500 MB",
    retention="10 days",
    level="DEBUG"
)

app = FastAPI(
    title="Multimodal RAG System",
    description="A Retrieval-Augmented Generation system supporting text, images, and PDFs",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1", tags=["RAG Operations"])


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("Starting Multimodal RAG System...")
    settings.create_directories()
    logger.info("Directories created/verified")
    
    logger.info(f"Server running on {settings.api_host}:{settings.api_port}")
    logger.info(f"Vector store: {settings.chroma_persist_directory}")
    logger.info(f"Using CLIP model: {settings.clip_model}")
    logger.info(f"Using embedding model: {settings.embedding_model}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Multimodal RAG System...")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Multimodal RAG System API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="info"
    )