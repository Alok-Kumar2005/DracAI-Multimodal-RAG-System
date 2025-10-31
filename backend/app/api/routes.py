from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import List, Optional
import shutil
from pathlib import Path
from loguru import logger
import asyncio
import json
from datetime import datetime

from backend.app.models import (
    UploadResponse, QueryRequest, QueryResponse, 
    HealthResponse, RetrievedDocument
)
from backend.app.config import settings
from backend.app.services.document_processor import DocumentProcessor
from backend.app.services.vector_store import VectorStore
from backend.app.services.query_service import QueryService
from backend.app.utils.token_counter import count_tokens

document_processor = DocumentProcessor()
vector_store = VectorStore()
query_service = QueryService(vector_store)

router = APIRouter()
conversations_db = {}


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
async def upload_document(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """Upload and process a document."""
    try:
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)
        
        if file_size > settings.max_upload_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.max_upload_size} bytes"
            )
        
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


@router.post("/query")
async def query_documents(request: QueryRequest):
    """Query the RAG system with optional thread_id for conversation history."""
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        query_tokens = count_tokens(request.query)
        logger.info(f"Query tokens: {query_tokens}")

        TOKEN_LIMIT = settings.token_limit
        thread_id = getattr(request, 'thread_id', None)
        if not thread_id:
            import uuid
            thread_id = str(uuid.uuid4())

        conversation_tokens = sum(
            msg.get("token_count", 0)
            for msg in conversations_db.get(thread_id, {}).get("messages", [])
        )

        total_estimated_tokens = query_tokens + conversation_tokens
        if query_tokens > TOKEN_LIMIT:
            raise HTTPException(
                status_code=400,
                detail=f"Query too long ({query_tokens} tokens). Limit is {TOKEN_LIMIT}."
            )

        if total_estimated_tokens > TOKEN_LIMIT:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Conversation exceeds token limit "
                    f"({total_estimated_tokens} tokens). Limit is {TOKEN_LIMIT}."
                )
            )
        result = query_service.query(
            query=request.query,
            top_k=request.top_k,
            filter_metadata=request.filter_metadata,
            include_images=request.include_images
        )

        answer_tokens = count_tokens(result["answer"])
        logger.info(f"Answer tokens: {answer_tokens}")

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
        if thread_id not in conversations_db:
            conversations_db[thread_id] = {
                "thread_id": thread_id,
                "title": request.query[:50] + "..." if len(request.query) > 50 else request.query,
                "messages": [],
                "created_at": str(datetime.now())
            }

        conversations_db[thread_id]["messages"].extend([
            {
                "role": "user",
                "content": request.query,
                "timestamp": str(datetime.now()),
                "token_count": query_tokens
            },
            {
                "role": "assistant",
                "content": result["answer"],
                "timestamp": str(datetime.now()),
                "retrieved_documents": [doc.dict() for doc in retrieved_docs],
                "processing_time": result["processing_time"],
                "token_count": answer_tokens
            }
        ])
        response = QueryResponse(
            query=result["query"],
            answer=result["answer"],
            retrieved_documents=retrieved_docs,
            total_results=result["total_results"],
            processing_time=result["processing_time"]
        )

        response_dict = response.dict()
        response_dict["thread_id"] = thread_id
        response_dict["query_tokens"] = query_tokens
        response_dict["answer_tokens"] = answer_tokens
        response_dict["total_tokens"] = query_tokens + answer_tokens
        response_dict["conversation_tokens"] = total_estimated_tokens

        return response_dict

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@router.get("/conversations")
async def get_conversations():
    """Get all conversation threads."""
    try:
        conversations = [
            {
                "thread_id": conv["thread_id"],
                "title": conv["title"],
                "created_at": conv["created_at"],
                "message_count": len(conv["messages"])
            }
            for conv in conversations_db.values()
        ]
        return conversations
    except Exception as e:
        logger.error(f"Error fetching conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations/{thread_id}")
async def get_conversation(thread_id: str):
    """Get a specific conversation thread."""
    try:
        if thread_id not in conversations_db:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conversations_db[thread_id]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/conversations/{thread_id}")
async def delete_conversation(thread_id: str):
    """Delete a conversation thread."""
    try:
        if thread_id not in conversations_db:
            raise HTTPException(status_code=404, detail="Conversation not found")
        del conversations_db[thread_id]
        return {"message": f"Conversation {thread_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
            conversations_db.clear()
            return {"message": "Database reset successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to reset database")
    except Exception as e:
        logger.error(f"Error resetting database: {e}")
        raise HTTPException(status_code=500, detail=str(e))