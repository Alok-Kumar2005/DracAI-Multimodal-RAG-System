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

