from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import List, Optional
import shutil
from pathlib import Path
from loguru import logger
import asyncio

from backend.app.models import (
    UploadResponse, QueryRequest, QueryResponse, 
    HealthResponse, RetrievedDocument
)
from backend.app.config import settings
from backend.app.services.document_processor import DocumentProcessor
from backend.app.services.vector_store import VectorStore
from backend.app.services.query_service import QueryService

document_processor = DocumentProcessor()
vector_store = VectorStore()
query_service = QueryService(vector_store)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        stats = vector_store.get_collection_stats()
        return HealthResponse(
            status="healthy",
            vector_store_status="connected",
            total_documents=stats["total_chunks"]
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...), background_tasks: BackgroundTasks = None ):
    """Upload and process a document."""
    try:
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)
        
        if file_size > settings.max_upload_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.max_upload_size} bytes"
            )
        
        ### first find the file type
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext in {'.txt', '.md', '.csv'}:
            save_dir = Path(settings.upload_directory) / "documents"
        elif file_ext in {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}:
            save_dir = Path(settings.upload_directory) / "images"
        elif file_ext == '.pdf':
            save_dir = Path(settings.upload_directory) / "pdfs"
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_ext}"
            )
        
        ### saving file
        save_dir.mkdir(parents=True, exist_ok=True)
        file_path = save_dir / file.filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Saved file: {file_path}")
        
        chunks, metadata = document_processor.process_document(str(file_path))
        document_id = document_processor.generate_document_id(str(file_path))
        chunks_created = vector_store.add_documents(chunks, document_id)
        return UploadResponse(
            success=True,
            message=f"Document processed successfully",
            document_id=document_id,
            metadata=metadata,
            chunks_created=chunks_created
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/upload/batch")
async def upload_batch(files: List[UploadFile] = File(...)):
    """Upload multiple documents at once."""
    results = []
    
    for file in files:
        try:
            result = await upload_document(file)
            results.append({
                "filename": file.filename,
                "success": True,
                "document_id": result.document_id,
                "chunks_created": result.chunks_created
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {"results": results, "total_files": len(files)}


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the RAG system."""
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        result = query_service.query(
            query=request.query,
            top_k=request.top_k,
            filter_metadata=request.filter_metadata,
            include_images=request.include_images
        )
        retrieved_docs = [
            RetrievedDocument(
                content=doc["content"],
                metadata=doc["metadata"],
                relevance_score=doc["relevance_score"],
                document_id=doc["document_id"],
                chunk_id=doc.get("chunk_id")
            )
            for doc in result["retrieved_documents"]
        ]
        
        return QueryResponse(
            query=result["query"],
            answer=result["answer"],
            retrieved_documents=retrieved_docs,
            total_results=result["total_results"],
            processing_time=result["processing_time"]
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@router.get("/documents")
async def list_documents():
    """List all documents in the system."""
    try:
        stats = vector_store.get_collection_stats()
        return {
            "total_chunks": stats["total_chunks"],
            "collection_name": stats["collection_name"]
        }
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and all its chunks."""
    try:
        success = vector_store.delete_document(document_id)
        if success:
            return {"message": f"Document {document_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset")
async def reset_database():
    """Reset the entire vector database (use with caution)."""
    try:
        success = vector_store.reset_collection()
        if success:
            return {"message": "Database reset successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to reset database")
    except Exception as e:
        logger.error(f"Error resetting database: {e}")
        raise HTTPException(status_code=500, detail=str(e))