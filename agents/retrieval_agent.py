from typing import List, Dict
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

class RetrievalAgent:
    def __init__(self, persist_directory: str = "./chroma_storage"):
        self.client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            persist_directory=persist_directory
        ))
        self.collection = self.client.get_or_create_collection(name="document_chunks")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def add_chunks(self, chunks: List[Dict]):
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

    def delete_file(self, source_file: str):
        self.collection.delete(where={"source_file": source_file})

    def list_files(self) -> List[str]:
        result = self.collection.get(include=["metadatas"])
        return list({meta["source_file"] for meta in result["metadatas"]})

    def query(self, user_query: str, top_k: int = 5) -> List[Dict]:
        embedding = self.embedder.encode([user_query])[0].tolist()
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["metadatas", "documents"]
        )
        return [
            {"text": doc, "metadata": meta}
            for doc, meta in zip(result["documents"][0], result["metadatas"][0])
        ]
