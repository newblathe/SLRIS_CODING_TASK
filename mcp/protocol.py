from typing import TypedDict, Literal

# Define the allowed message types for structured communication
MessageType = Literal[
    "INGESTION_REQUEST",   # File upload and ingestion request
    "ADDTO_DB",            # Add processed chunks to vector DB
    "DELETEFROM_DB",       # Delete vector DB entries for a given file
    "RETRIEVAL_REQUEST",   # Request for chunk retrieval based on query
    "LLM_REQUEST",         # Forward query and chunks to LLM
    "LLM_RESPONSE"         # Response from LLM to user
]

class MCPMessage(TypedDict):
    """
    Message structure for the Model Communication Protocol (MCP).

    Fields:
        sender (str): Name of the agent sending the message (e.g., "UI", "Coordinator").
        receiver (str): Target agent for the message (e.g., "IngestionAgent", "LLMResponseAgent").
        type (MessageType): Type of message being sent (defined above).
        trace_id (str): Unique identifier for tracking the full lifecycle of a message/query.
        payload (dict): Payload content, specific to the message type.
    """
    sender: str
    receiver: str
    type: MessageType
    trace_id: str
    payload: dict