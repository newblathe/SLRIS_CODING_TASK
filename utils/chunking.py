import nltk
from nltk.tokenize import sent_tokenize
from typing import List, Dict
from utils.file_parser import parse_file
import os
nltk.download("punkt")

def chunk_text_data(parsed_data: List[Dict]) -> List[Dict]:
    """
    Splits text entries into sentences and retains metadata.
    Table rows are returned as-is.
    """
    chunks = []

    for entry in parsed_data:
        if entry["type"] in {"table_row", "csv_row"}:
            # Keep table data unchunked
            chunks.append(entry)
            continue

        sentences = sent_tokenize(entry["text"])
        for i, sentence in enumerate(sentences):
            chunk = {
                "text": sentence,
                "source_file": entry["source_file"],
                "sentence": i+1,
                "type": "sentence",
            }

            # Preserve contextual metadata
            for key in ["page", "paragraph", "slide"]:
                if key in entry:
                    chunk[key] = entry[key]

            chunks.append(chunk)

    return chunks