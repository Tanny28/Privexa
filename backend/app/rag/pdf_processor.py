# ──────────────────────────────────────────────
# PDF Text Extraction
# ──────────────────────────────────────────────
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict


def extract_text_from_pdf(file_path: str) -> List[Dict]:
    """
    Extract text from a PDF file, page by page.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of dicts with 'page_number' and 'text' keys.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF file: {file_path}")

    pages = []
    doc = fitz.open(str(path))

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")  # plain text extraction

        if text.strip():  # skip empty pages
            pages.append({
                "page_number": page_num + 1,
                "text": text.strip(),
            })

    doc.close()
    return pages


def extract_full_text(file_path: str) -> str:
    """
    Extract all text from a PDF as a single string.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Combined text from all pages.
    """
    pages = extract_text_from_pdf(file_path)
    return "\n\n".join(p["text"] for p in pages)


def get_pdf_metadata(file_path: str) -> Dict:
    """
    Extract metadata from a PDF file.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Dict with metadata (title, author, page_count, etc.)
    """
    doc = fitz.open(file_path)
    metadata = doc.metadata
    page_count = len(doc)
    doc.close()

    return {
        "title": metadata.get("title", ""),
        "author": metadata.get("author", ""),
        "subject": metadata.get("subject", ""),
        "page_count": page_count,
    }
