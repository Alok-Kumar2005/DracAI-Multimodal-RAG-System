import fitz
from PIL import Image
import io
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Any
from loguru import logger
import base64
from datetime import datetime

from backend.app.config import settings
from backend.app.models import FileType, DocumentMetadata


class DocumentProcessor:
    """Processing various document types into chunks with metadata."""
    def __init__(self):
        self.supported_image_formats = {'.png','.jpg', '.jpeg','.gif', '.bmp'}
        self.supported_text_formats = {'.txt', '.md','.csv'}
        self.supported_pdf_format = {'.pdf'}
    
    def generate_document_id(self, file_path: str) -> str:
        """Generating a unique document ID based on file content."""
        with open(file_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        return file_hash[:16]
    
    def get_file_type(self, file_path: str) -> FileType:
        """Getting file type based on extension."""
        ext = Path(file_path).suffix.lower()
        if ext in self.supported_image_formats:
            return FileType.IMAGE
        elif ext in self.supported_text_formats:
            return FileType.TEXT
        elif ext in self.supported_pdf_format:
            return FileType.PDF
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    def process_text_file(self, file_path: str) -> Tuple[List[Dict[str, Any]], DocumentMetadata]:
        """Process plain text files."""
        logger.info(f"Processing text file: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        file_stat = Path(file_path).stat()
        
        metadata = DocumentMetadata(
            file_name=Path(file_path).name,
            file_type=FileType.TEXT,
            file_size=file_stat.st_size,
            has_text=True,
            has_images=False,
            source_path=file_path
        )
        chunks = self._chunk_text(content, metadata.model_dump())
        return chunks, metadata
    
    def process_image_file(self, file_path: str) -> Tuple[List[Dict[str, Any]], DocumentMetadata]:
        """Process image files."""
        logger.info(f"Processing image file: {file_path}")
        
        # Load image
        image = Image.open(file_path)
        file_stat = Path(file_path).stat()
        
        # Convert image to base64 for storage
        buffered = io.BytesIO()
        image.save(buffered, format=image.format or "PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        metadata = DocumentMetadata(
            file_name=Path(file_path).name,
            file_type=FileType.IMAGE,
            file_size=file_stat.st_size,
            has_text=False,
            has_images=True,
            source_path=file_path
        )
        
        # Create single chunk for image
        chunks = [{
            "content": f"Image: {Path(file_path).name}",
            "image_data": img_base64,
            "metadata": metadata.model_dump(),
            "chunk_type": "image"
        }]
        
        return chunks, metadata
    
    def process_pdf_file(self, file_path: str) -> Tuple[List[Dict[str, Any]], DocumentMetadata]:
        """Process PDF files with text and images."""
        logger.info(f"Processing PDF file: {file_path}")
        
        doc = fitz.open(file_path)
        chunks = []
        has_text = False
        has_images = False
        page_count = len(doc) 
        
        for page_num in range(page_count):
            page = doc[page_num]
            
            # Extract text
            text = page.get_text()
            if text.strip():
                has_text = True
                page_metadata = {
                    "file_name": Path(file_path).name,
                    "page_number": page_num + 1,
                    "chunk_type": "text",
                    "source_path": file_path
                }
                text_chunks = self._chunk_text(text, page_metadata)
                chunks.extend(text_chunks)
            
            # Extract images
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                has_images = True
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Convert to PIL Image
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    # Convert to base64
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode()
                    
                    chunks.append({
                        "content": f"Image from {Path(file_path).name}, Page {page_num + 1}, Image {img_index + 1}",
                        "image_data": img_base64,
                        "metadata": {
                            "file_name": Path(file_path).name,
                            "page_number": page_num + 1,
                            "image_index": img_index + 1,
                            "chunk_type": "image",
                            "source_path": file_path
                        },
                        "chunk_type": "image"
                    })
                except Exception as e:
                    logger.warning(f"Failed to extract image {img_index + 1} from page {page_num + 1}: {e}")
                    continue
        
        doc.close()
        
        file_stat = Path(file_path).stat()
        file_type = FileType.MIXED if (has_text and has_images) else (FileType.TEXT if has_text else FileType.IMAGE)
        
        metadata = DocumentMetadata(
            file_name=Path(file_path).name,
            file_type=file_type,
            file_size=file_stat.st_size,
            page_count=page_count,  
            has_text=has_text,
            has_images=has_images,
            source_path=file_path
        )
        
        return chunks, metadata
    
    def _chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text into chunks with overlap."""
        chunks = []
        chunk_size = settings.chunk_size
        chunk_overlap = settings.chunk_overlap
        
        # Simple character-based chunking
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            
            if chunk_text.strip():
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_id"] = chunk_id
                chunk_metadata["chunk_type"] = "text"
                
                chunks.append({
                    "content": chunk_text,
                    "metadata": chunk_metadata,
                    "chunk_type": "text"
                })
                chunk_id += 1
            
            start += chunk_size - chunk_overlap
        
        return chunks
    
    def process_document(self, file_path: str) -> Tuple[List[Dict[str, Any]], DocumentMetadata]:
        """Main method to process any supported document type."""
        file_type = self.get_file_type(file_path)
        
        if file_type == FileType.TEXT:
            return self.process_text_file(file_path)
        elif file_type == FileType.IMAGE:
            return self.process_image_file(file_path)
        elif file_type == FileType.PDF:
            return self.process_pdf_file(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")