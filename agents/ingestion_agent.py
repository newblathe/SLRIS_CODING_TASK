from typing import List, Dict
from utils.file_parser import parse_file
from utils.chunking import chunk_text_data

class IngestionAgent:
    """
    Parses and preprocesses a document.
    Returns sentence-level chunks with metadata.
    """

    def __init__(self):
        pass  # No need to initialize anything for parsing

    def process_file(self, file_path: str) -> List[Dict]:
        """
        Parse and chunk a file into sentences with metadata.
        """
        parsed_elements = parse_file(file_path)
        chunks = chunk_text_data(parsed_elements)
        return chunks