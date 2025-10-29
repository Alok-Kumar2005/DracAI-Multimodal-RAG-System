import torch
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from sentence_transformers import SentenceTransformer
from PIL import Image
import io
import base64
from typing import List, Union
from loguru import logger
import numpy as np

from backend.app.config import settings


class EmbeddingService:
    """Generating Embedding of images and text using clip model"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  ## Using cpu for compatibility
        logger.info(f"Using device: {self.device}")
        
        ### Clip model loading
        logger.info(f"Loading CLIP model: {settings.clip_model}")
        self.clip_model = CLIPModel.from_pretrained(settings.clip_model).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(settings.clip_model)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(settings.clip_model)
        
        ### sentence transformer for text embeddings
        logger.info(f"Loading text embedding model: {settings.embedding_model}")
        self.text_model = SentenceTransformer(settings.embedding_model)
        
        self.embedding_dimension = 512 
    
    def embed_text(self, text: str) -> List[float]:
        """Embedding fof text using CLIP model."""
        try:
            inputs = self.clip_tokenizer(
                text, 
                padding=True, 
                truncation=True, 
                return_tensors="pt",
                max_length=77
            ).to(self.device)
            
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)
                ### normalize embeddings in 0-1 range
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            embedding = text_features.cpu().numpy().flatten().tolist()
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating text embedding: {e}")
            ### if clipp model fails then Setnce transformer as fallback
            embedding = self.text_model.encode(text).tolist()
            if len(embedding) < self.embedding_dimension:
                embedding.extend([0.0] * (self.embedding_dimension - len(embedding)))
            else:
                embedding = embedding[:self.embedding_dimension]
            return embedding
    
    def embed_image(self, image_data: str) -> List[float]:
        """Embeddings for images using CLIP model."""
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            embedding = image_features.cpu().numpy().flatten().tolist()
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating image embedding: {e}")
            ### return 0 vector if embedding fails
            return [0.0] * self.embedding_dimension
    
    def embed_batch_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_text(text))
        return embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for search query."""
        return self.embed_text(query)
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings."""
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)