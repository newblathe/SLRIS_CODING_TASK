from typing import Dict
from groq import Groq
import json
import re
from datetime import datetime
import os

from dotenv import load_dotenv

from mcp.protocol import MCPMessage

# Load environment variables (specifically the GROQ API key)
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)

def format_citation(meta: Dict) -> str:
    """
    Formats citation metadata into a human-readable string.

    Args:
        meta (Dict): Metadata containing fields like page, paragraph, slide, table, etc.

    Returns:
        str: Formatted citation string.
    """
    parts = []
    if meta.get("page") is not None:
        parts.append(f"Page {meta['page']}")
    if meta.get("paragraph") is not None:
        parts.append(f"Para {meta['paragraph']}")
    if meta.get("slide") is not None:
        parts.append(f"Slide {meta['slide']}")
    if meta.get("table") is not None:
        parts.append(f"Table {meta['table']}")
    if meta.get("row") is not None:
        parts.append(f"Row {meta['row']}")
    if meta.get("source_file") is not None:
        parts.append(f"Source: {meta['source_file']}")
    return ", ".join(parts) if parts else "Unknown"

class LLMResponseAgent:
    """
    Agent responsible for synthesizing a final response from the top-k retrieved chunks 
    using the LLM hosted via Groq API.
    """
    def __init__(self):
        """
        Initializes the LLM configuration.
        """
        self.model = "llama-3.3-70b-versatile"

    def generate(self, message: MCPMessage) -> MCPMessage:
        """
        Constructs a prompt using the user query and top retrieved chunks, 
        and sends it to the LLM to generate an answer.

        Args:
            message (MCPMessage): Message of type 'LLM_REQUEST' containing query and chunks.

        Returns:
            MCPMessage: LLM response wrapped in a MCPMessage with source citations.
        """
        assert message["type"] == "LLM_REQUEST"

        user_query = message["payload"]["user_query"]
        top_chunks = message["payload"]["top_chunks"]

        # Format chunks with citation metadata
        formatted_chunks = []
        for i, chunk in enumerate(top_chunks):
            citation = format_citation(chunk.get("metadata", {}))
            formatted_chunks.append(f"[Chunk {i+1} | Citation: {citation}], text: {chunk['text'].strip()}")

        # Build prompt by injecting top chunks and the user query
        combined_chunks = "\n".join(formatted_chunks)
        prompt = f"""
You are a factual AI assistant. Given the following document chunks with the citations and text and a user question, perform two tasks:

1. Extract a complete, accurate answer using the original terms, numbers, and clauses.
2. Provide the citation by referencing the provided citations.

Chunks:
{combined_chunks}

Question:
{user_query}

Return only JSON in one line, e.g.:
{{"answer":"...", "citation":"Page 2, Para 4, Source: abc.pdf"}}
"""

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            content = response.choices[0].message.content.strip()
            match = re.search(r'{.*}', content)
            if not match:
                raise ValueError("Invalid response format")

            parsed = json.loads(match.group())
            return MCPMessage(
                sender="LLMResponseAgent",
                receiver="CoordinatorAgent",
                type="LLM_RESPONSE",
                trace_id= message["trace_id"],
                payload={
                    "response": parsed.get("answer", ""),
                    "citation": parsed.get("citation", "Unknown"),
                    "query": user_query,
                }
            )

        except Exception as e:
            return MCPMessage(
                sender="LLMResponseAgent",
                receiver="CoordinatorAgent",
                type="LLM_RESPONSE",
                trace_id= message["trace_id"],
                payload={
                    "response": f"Error: {str(e)}",
                    "citation": "Unknown",
                    "query": user_query,
                }
            )
