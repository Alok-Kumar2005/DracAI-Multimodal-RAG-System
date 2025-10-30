from typing_extensions import TypedDict
from typing import List, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


class RAGState(TypedDict):
    query: str
    retrieved_docs: List[Dict[str, Any]]
    context: str
    answer: str
    metadata: Dict[str, Any]
    messages: Annotated[List[BaseMessage], "conversation messages"]