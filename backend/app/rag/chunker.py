# ──────────────────────────────────────────────
# Text Chunking
# ──────────────────────────────────────────────
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: The full text to chunk.
        chunk_size: Maximum size of each chunk (characters).
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def chunk_pages(
    pages: List[Dict],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Dict]:
    """
    Chunk page-level text while preserving page metadata.

    Args:
        pages: List of dicts with 'page_number' and 'text'.
        chunk_size: Maximum size of each chunk.
        chunk_overlap: Overlap between chunks.

    Returns:
        List of dicts with 'text', 'page_number', and 'chunk_index'.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    chunk_index = 0

    for page in pages:
        page_chunks = splitter.split_text(page["text"])
        for chunk_text_str in page_chunks:
            all_chunks.append({
                "text": chunk_text_str,
                "page_number": page["page_number"],
                "chunk_index": chunk_index,
            })
            chunk_index += 1

    return all_chunks
