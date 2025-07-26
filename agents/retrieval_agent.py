import chromadb
from typing import List
from mcp.protocol import MCPMessage
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

class RetrievalAgent:
    """
    Agent responsible for storing document chunks in a vector store (ChromaDB),
    retrieving relevant chunks for a given query, and deleting documents.
    """
    def __init__(self, persist_directory: str = "./chroma_storage"):
        """
        Initializes the persistent ChromaDB client and embedding model.

        Args:
            persist_directory (str): Path where ChromaDB storage will persist.
        """
        self.client = PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name="document_chunks")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def add_chunks(self, message: MCPMessage) -> str:
        """
        Adds parsed and chunked documents to ChromaDB for future retrieval.

        Args:
            message (MCPMessage): Message with type "ADDTO_DB" containing list of chunks.

        Returns:
            str: Status message confirming successful storage.
        """

        assert message["type"] == "ADDTO_DB"
        chunks = message["payload"]["chunks"]
        texts = [chunk["text"] for chunk in chunks]
        ids = [f"{chunk['source_file']}_{i}" for i, chunk in enumerate(chunks)]
        metadatas = []

        for chunk in chunks:
            metadata = {"source_file": chunk["source_file"], "type": chunk["type"]}
            for key in ["page", "paragraph", "slide", "row", "table", "sentence"]:
                if chunk.get(key) is not None:
                    metadata[key] = chunk[key]
            metadatas.append(metadata)

        embeddings = self.embedder.encode(texts).tolist()

        self.collection.upsert(documents=texts, ids=ids, metadatas=metadatas, embeddings=embeddings)

        return "Added to ChromeDB"

    def delete_file(self, message: MCPMessage) -> str:
        """
        Deletes all chunks related to a specific file from ChromaDB.

        Args:
            message (MCPMessage): Message of type "DELETEFROM_DB" containing filename.

        Returns:
            str: Status message confirming deletion.
        """
        assert message["type"] == "DELETEFROM_DB"
        source_file = message["payload"]["source_file"]

        # Filter and delete by source_file metadata
        self.collection.delete(where={"source_file": source_file})
        return f"Chunks for file {source_file} deleted from ChromaDB"

    def list_files(self) -> List[str]:
        """
        Lists all files currently stored in the local chromadb.

        Returns:
            list[str]: A sorted list of filenames present in the chromadb.
        """
        result = self.collection.get(include=["metadatas"])
        return list({meta["source_file"] for meta in result["metadatas"]})

    def query(self, message: MCPMessage, top_k: int = 5) -> MCPMessage:
        """
        Performs semantic search on ChromaDB and returns top-k relevant chunks.

        Args:
            message (MCPMessage): Message of type "RETRIEVAL_REQUEST" with query.

        Returns:
            MCPMessage: Message to be passed to LLM with top-k chunks and user query.
        """

        assert message["type"] == "RETRIEVAL_REQUEST"
        user_query = message["payload"]["user_query"]
        embedding = self.embedder.encode([user_query])[0].tolist()
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["metadatas", "documents"]
        )
        return MCPMessage(
            sender= "RetrievalAgent",
            receiver= "LLMResponseAgent",
            type= "LLM_REQUEST",
            trace_id= message["trace_id"],
            payload= {"top_chunks": [{"text": doc, "metadata": meta} for doc, meta in zip(result["documents"][0], result["metadatas"][0])],
                      "user_query": user_query}
        )
