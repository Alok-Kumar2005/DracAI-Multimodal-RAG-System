from typing import List, Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver
import aiosqlite
import uuid
from loguru import logger
import time

from backend.app.config import settings
from backend.app.graph.state import RAGState
from backend.app.graph.prompt import system_prompt
from backend.app.services.vector_store import VectorStore

class QueryService:
    """Service for handling queries with LangChain and LangGraph."""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.7,
            google_api_key=settings.gemini_api_key
        )
        self.checkpointer = None
        self.rag_graph = None

    async def initialize(self):
        self.checkpointer = AsyncSqliteSaver.from_conn_string("checkpoints.db")
        self.rag_graph = self._build_rag_graph()
    
    def _build_rag_graph(self) -> StateGraph:
        workflow = StateGraph(RAGState)
        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("generate", self._generate_answer)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        return workflow.compile(checkpointer=self.checkpointer)
    
    def _retrieve_documents(self, state: RAGState) -> RAGState:
        """Retrieve relevant documents from vector store."""
        logger.info(f"Retrieving documents for query: {state['query']}")
        
        ### get metadata if present
        metadata_filter = state.get("metadata", {}).get("filter", None)
        include_images = state.get("metadata", {}).get("include_images", True)
        top_k = state.get("metadata", {}).get("top_k", settings.top_k_results)
        
        ### document retrieval
        retrieved_docs = self.vector_store.query(
            query_text=state["query"],
            top_k=top_k,
            filter_dict=metadata_filter,
            include_images=include_images
        )
        
        ### context building fron search document
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            doc_type = doc["metadata"].get("chunk_type", "text")
            source = doc["metadata"].get("file_name", "Unknown")
            
            if doc_type == "image":
                context_parts.append(
                    f"[Document {i}] (Image from {source})\n"
                    f"Description: {doc['content']}\n"
                    f"Relevance Score: {doc['relevance_score']:.2f}\n"
                )
            else:
                context_parts.append(
                    f"[Document {i}] (Text from {source})\n"
                    f"Content: {doc['content']}\n"
                    f"Relevance Score: {doc['relevance_score']:.2f}\n"
                )
        
        context = "\n---\n".join(context_parts)
        
        state["retrieved_docs"] = retrieved_docs
        state["context"] = context
        
        return state
    
    def _generate_answer(self, state: RAGState) -> RAGState:
        """Generate answer using LLM based on retrieved context."""
        logger.info("Generating answer with LLM")
        
        if not state["retrieved_docs"]:
            state["answer"] = "I couldn't find any relevant information..."
            return state
        
        conversation_history = state.get("messages", [])
        
        user_message = f"""Context from knowledge base:
{state['context']}
Question: {state['query']}
Please provide a comprehensive answer based on the context above."""
        
        try:
            # Build messages with history
            messages = [SystemMessage(content=system_prompt)]
            messages.extend(conversation_history[-10:])
            messages.append(HumanMessage(content=user_message))
            response = self.llm.invoke(messages)
            state["answer"] = response.content
            state["messages"] = conversation_history + [
                HumanMessage(content=state["query"]),
                AIMessage(content=response.content)
            ]
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            state["answer"] = f"Error: {str(e)}"
            state["messages"] = conversation_history
        
        return state

    
    def query(self, query: str, top_k: int = 5,filter_metadata: Optional[Dict[str, Any]] = None,include_images: bool = True) -> Dict[str, Any]:
        """Execute a query through the RAG pipeline."""
        start_time = time.time()
        initial_state: RAGState = {
            "query": query,
            "retrieved_docs": [],
            "context": "",
            "answer": "",
            "metadata": {
                "top_k": top_k,
                "filter": filter_metadata,
                "include_images": include_images
            }
        }
        
        try:
            result = self.rag_graph.invoke(initial_state)
            processing_time = time.time() - start_time
            
            return {
                "query": query,
                "answer": result["answer"],
                "retrieved_documents": result["retrieved_docs"],
                "total_results": len(result["retrieved_docs"]),
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error in query pipeline: {e}")
            return {
                "query": query,
                "answer": f"An error occurred while processing your query: {str(e)}",
                "retrieved_documents": [],
                "total_results": 0,
                "processing_time": time.time() - start_time
            }
    
    def query_with_conversation_history( self, query: str, conversation_history: List[Dict[str, str]], top_k: int = 5 ) -> Dict[str, Any]:
        """Query with conversation context."""
        if conversation_history:
            history_text = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in conversation_history[-3:]  ### last 3 truns
            ])
            
            reformulation_prompt = f"""Given the conversation history:
{history_text}
And the new question: {query}
Reformulate the question to be standalone and incorporate relevant context from the history.
Provide only the reformulated question, nothing else."""
            
            try:
                messages = [HumanMessage(content=reformulation_prompt)]
                response = self.llm.invoke(messages)
                reformulated_query = response.content.strip()
                logger.info(f"Reformulated query: {reformulated_query}")
            except:
                reformulated_query = query
        else:
            reformulated_query = query
        return self.query(reformulated_query, top_k=top_k)