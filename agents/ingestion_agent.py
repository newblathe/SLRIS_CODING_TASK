from mcp.protocol import MCPMessage
from utils.file_parser import parse_file
from utils.chunking import chunk_text_data

class IngestionAgent:
    """
    Agent responsible for preprocessing documents before ingestion.
    It handles file parsing and chunking into semantically meaningful units.
    """

    def preprocess(self, message: MCPMessage) -> MCPMessage:
        """
        Parses and chunks the files provided in the MCP message payload.

        Args:
            message (MCPMessage): Message with type "INGESTION_REQUEST" and payload containing file paths.

        Returns:
            MCPMessage: New message of type "ADDTO_DB" with all generated chunks and metadata.
        """
        assert message["type"] == "INGESTION_REQUEST", "Invalid message type for ingestion"

        file_paths = message["payload"]["file_paths"]
        all_chunks = []

        for path in file_paths:
            parsed_file = parse_file(path)         # Extracts structured content from the file
            chunks = chunk_text_data(parsed_file)  # Converts parsed content into semantic chunks
            all_chunks.extend(chunks)

        return MCPMessage(
            sender="IngestionAgent",
            receiver="RetrievalAgent",
            type="ADDTO_DB",
            trace_id=message["trace_id"],
            payload={"chunks": all_chunks}
        )