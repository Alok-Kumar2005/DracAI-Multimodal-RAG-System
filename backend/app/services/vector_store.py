import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional
from loguru import logger
import uuid

from backend.app.config import settings
from backend.app.services.embedding_service import EmbeddingService

# class VectorStore:
#     """Vector store of text and images using ChromaDB"""
    
#     def __init__(self):
#         self.embedding_service = EmbeddingService()
#         self.client = chromadb.PersistentClient(
#             path=settings.chroma_persist_directory,
#             settings=ChromaSettings(
#                 anonymized_telemetry=False,
#                 allow_reset=True
#             )
#         )
        
#         ### getting existing collection or creating new one if not found
#         try:
#             self.collection = self.client.get_collection(
#                 name=settings.collection_name
#             )
#             logger.info(f"Loaded existing collection: {settings.collection_name}")
#         except:
#             self.collection = self.client.create_collection(
#                 name=settings.collection_name,
#                 metadata={"description": "Multimodal RAG collection"}
#             )
#             logger.info(f"Created new collection: {settings.collection_name}")
    
#     def add_documents(self, chunks: List[Dict[str, Any]], document_id: str) -> int:
#         """Add document chunks to the vector store."""
#         ids = []
#         embeddings = []
#         documents = []
#         metadatas = []
        
#         for i, chunk in enumerate(chunks):
#             chunk_id = f"{document_id}_{i}"
#             ids.append(chunk_id)
#             if chunk.get("chunk_type") == "image":
#                 embedding = self.embedding_service.embed_image(chunk["image_data"])
#                 ### sstoring refernces as metadata in images
#                 chunk["metadata"]["has_image"] = True
#                 chunk["metadata"]["image_id"] = chunk_id
#             else:
#                 embedding = self.embedding_service.embed_text(chunk["content"])
#                 chunk["metadata"]["has_image"] = False
            
#             embeddings.append(embedding)
#             documents.append(chunk["content"])
#             metadata = chunk["metadata"].copy()
#             metadata["document_id"] = document_id
#             metadata["chunk_type"] = chunk.get("chunk_type", "text")
            
#             ## datetime to string ( if present )
#             if "upload_timestamp" in metadata:
#                 metadata["upload_timestamp"] = str(metadata["upload_timestamp"])
#             metadatas.append(metadata)
        
#         # Add to ChromaDB
#         try:
#             self.collection.add(ids=ids, embeddings=embeddings,
#                 documents=documents, metadatas=metadatas
#             )
#             logger.info(f"Added {len(chunks)} chunks for document {document_id}")
#             return len(chunks)
#         except Exception as e:
#             logger.error(f"Error adding documents to vector store: {e}")
#             raise
    
#     def query(self, query_text: str, top_k: int = 5,filter_dict: Optional[Dict[str, Any]] = None,include_images: bool = True) -> List[Dict[str, Any]]:
#         """Query the vector store."""
#         query_embedding = self.embedding_service.embed_query(query_text)
#         # Build where clause for filtering
#         where_clause = None
#         if filter_dict:
#             where_clause = filter_dict
        
#         # If not including images, filter them out
#         if not include_images:
#             if where_clause:
#                 where_clause["chunk_type"] = {"$ne": "image"}
#             else:
#                 where_clause = {"chunk_type": {"$ne": "image"}}
        
#         ### relavent documents from database
#         try:
#             results = self.collection.query(
#                 query_embeddings=[query_embedding],
#                 n_results=top_k,
#                 where=where_clause,
#                 include=["documents", "metadatas", "distances"]
#             )
            
#             # Format results
#             retrieved_docs = []
#             if results["ids"][0]:  # Check if we got any results
#                 for i in range(len(results["ids"][0])):
#                     doc = {
#                         "content": results["documents"][0][i],
#                         "metadata": results["metadatas"][0][i],
#                         "relevance_score": 1.0 - float(results["distances"][0][i]),
#                         "chunk_id": results["ids"][0][i],
#                         "document_id": results["metadatas"][0][i].get("document_id", "unknown")
#                     }
#                     retrieved_docs.append(doc)
            
#             logger.info(f"Retrieved {len(retrieved_docs)} documents for query: {query_text[:50]}...")
#             return retrieved_docs
            
#         except Exception as e:
#             logger.error(f"Error querying vector store: {e}")
#             return []
    
#     def delete_document(self, document_id: str) -> bool:
#         """Delete all chunks belonging to a document."""
#         try:
#             results = self.collection.get(
#                 where={"document_id": document_id},
#                 include=[]
#             )
            
#             if results["ids"]:
#                 self.collection.delete(ids=results["ids"])
#                 logger.info(f"Deleted document {document_id} ({len(results['ids'])} chunks)")
#                 return True
#             return False
            
#         except Exception as e:
#             logger.error(f"Error deleting document: {e}")
#             return False
    
#     def get_collection_stats(self) -> Dict[str, Any]:
#         """Get statistics about the collection."""
#         try:
#             count = self.collection.count()
#             return {
#                 "total_chunks": count,
#                 "collection_name": settings.collection_name
#             }
#         except Exception as e:
#             logger.error(f"Error getting collection stats: {e}")
#             return {"total_chunks": 0, "collection_name": settings.collection_name}
    
#     def reset_collection(self) -> bool:
#         """Reset the entire collection (use with caution)."""
#         try:
#             self.client.delete_collection(name=settings.collection_name)
#             self.collection = self.client.create_collection(
#                 name=settings.collection_name,
#                 metadata={"description": "Multimodal RAG collection"}
#             )
#             logger.warning("Collection reset complete")
#             return True
#         except Exception as e:
#             logger.error(f"Error resetting collection: {e}")
#             return False