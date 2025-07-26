from typing import Optional

from mcp.protocol import MCPMessage
from agents.ingestion_agent import IngestionAgent
from agents.retrieval_agent import RetrievalAgent
from agents.llmresponse_agent import LLMResponseAgent


class CoordinatorAgent:
    """
    Orchestrates the processing of user requests by routing messages to the 
    appropriate internal agents: IngestionAgent, RetrievalAgent, and LLMResponseAgent.
    """

    def __init__(self):
        """
        Initializes all internal agents used by the coordinator.
        """
        self.ingestor = IngestionAgent()
        self.retriever = RetrievalAgent()
        self.llm = LLMResponseAgent()

    def handle_message(self, message: MCPMessage) -> Optional[MCPMessage]:
        """
        Main dispatcher for handling incoming MCP messages.
        Routes to the correct agent based on message type.

        Args:
            message (MCPMessage): The message to handle, including type and payload.

        Returns:
            Optional[MCPMessage]: The response message after handling, or None if unknown type.
        """
        message_type = message.get("type")

        if message_type == "INGESTION_REQUEST":
            # Preprocess document and add its chunks to the vector store
            parsed_message = self.ingestor.preprocess(message)
            response = self.retriever.add_chunks(parsed_message)
            return response

        elif message_type == "DELETEFROM_DB":
            # Delete the file and its related chunks from the vector store
            response = self.retriever.delete_file(message)
            return response

        elif message_type == "RETRIEVAL_REQUEST":
            # Retrieve relevant chunks and generate an LLM-based answer
            retrieval_response = self.retriever.query(message)
            llm_response = self.llm.generate(retrieval_response)
            return llm_response

        else:
            # Unknown or unsupported message type
            print(f"Unknown message type: {message_type}")
            return None
