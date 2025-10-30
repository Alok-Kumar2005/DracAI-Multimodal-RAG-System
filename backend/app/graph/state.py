from typing import List, Dict, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing import Annotated


class RAGState(TypedDict):
    query: str
    retrieved_docs: List[Dict[str, Any]]
    context: str
    answer: str
    metadata: Dict[str, Any]
    messages: Annotated[List[BaseMessage], add_messages]