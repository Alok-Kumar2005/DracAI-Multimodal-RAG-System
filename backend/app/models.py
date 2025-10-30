from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class FileType(str, Enum):
    """Supported file types."""
    TEXT = "text"
    IMAGE = "image"
    PDF = "pdf"
    MIXED = "mixed"


class DocumentMetadata(BaseModel):
    """Metadata for processed documents."""
    file_name: str
    file_type: FileType
    upload_timestamp: datetime = Field(default_factory=datetime.now)
    file_size: int
    page_count: Optional[int] = None
    has_images: bool = False
    has_text: bool = False
    source_path: str

class UploadResponse(BaseModel):
    """Response model for document upload."""
    success: bool
    message: str
    document_id: str
    metadata: DocumentMetadata
    chunks_created: int


class QueryRequest(BaseModel):
    """Request model for querying the RAG system."""
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    filter_metadata: Optional[Dict[str, Any]] = None
    include_images: bool = True


class RetrievedDocument(BaseModel):
    """Model for retrieved document chunks."""
    content: str
    metadata: Dict[str, Any]
    relevance_score: float
    document_id: str
    chunk_id: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for query results."""
    query: str
    answer: str
    retrieved_documents: List[RetrievedDocument]
    total_results: int
    processing_time: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime = Field(default_factory=datetime.now)
    vector_store_status: str
    total_documents: int
