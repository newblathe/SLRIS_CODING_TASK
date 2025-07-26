import nltk
from nltk.tokenize import sent_tokenize
from typing import List, Dict

# Ensure sentence tokenizer is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

def chunk_text_data(parsed_data: List[Dict]) -> List[Dict]:
    """
    Splits parsed text data into sentence-level chunks while preserving contextual metadata.

    This function handles two types of data:
    - Text entries: split into individual sentences.
    - Table rows: retained as-is (not split).

    Args:
        parsed_data (List[Dict]): A list of dictionaries containing parsed content from documents.

    Returns:
        List[Dict]: A list of chunked text entries, each with metadata including file name, type,
                    sentence index, and optionally page/paragraph/slide references.
    """
    chunks = []

    for entry in parsed_data:
        if entry["type"] in {"table_row", "csv_row"}:
            # Keep table rows or CSV rows unchanged
            chunks.append(entry)
            continue

        # Tokenize the text into sentences
        sentences = sent_tokenize(entry["text"])
        for i, sentence in enumerate(sentences):
            chunk = {
                "text": sentence,
                "source_file": entry["source_file"],
                "sentence": i + 1,
                "type": "sentence"
            }

            # Preserve context metadata
            for key in ["page", "paragraph", "slide"]:
                if key in entry:
                    chunk[key] = entry[key]

            chunks.append(chunk)

    return chunks