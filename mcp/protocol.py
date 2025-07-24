from typing import TypedDict, Literal, List, Optional, Dict

class Message(TypedDict):
    role: Literal["user", "agent", "coordinator"]
    content: str
    source_chunks: Optional[List[dict]]

class Task(TypedDict):
    id: str
    type: Literal["ingestion", "retrieval", "response"]
    input: Dict
    status: Literal["pending", "in_progress", "complete"]
    messages: List[Message]