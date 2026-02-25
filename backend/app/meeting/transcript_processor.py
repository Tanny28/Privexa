# ──────────────────────────────────────────────
# Transcript Processor — Read & clean transcripts
# ──────────────────────────────────────────────
from pathlib import Path
from typing import Optional


def read_transcript(file_path: str) -> str:
    """
    Read a transcript file from disk.

    Args:
        file_path: Path to the .txt transcript file.

    Returns:
        Cleaned transcript text.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Transcript not found: {file_path}")

    text = path.read_text(encoding="utf-8")
    return clean_transcript(text)


def clean_transcript(text: str) -> str:
    """
    Clean raw transcript text.
    Removes excessive whitespace, empty lines, and normalizes formatting.

    Args:
        text: Raw transcript text.

    Returns:
        Cleaned text.
    """
    lines = text.strip().splitlines()
    cleaned = []
    for line in lines:
        line = line.strip()
        if line:
            cleaned.append(line)
    return "\n".join(cleaned)


def chunk_transcript(text: str, max_chars: int = 3000) -> list[str]:
    """
    Split a long transcript into chunks that fit within LLM context.
    Splits on paragraph boundaries when possible.

    Args:
        text: Full transcript text.
        max_chars: Maximum characters per chunk.

    Returns:
        List of transcript chunks.
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    paragraphs = text.split("\n")
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 1 > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            current_chunk += "\n" + para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def estimate_duration(text: str, words_per_minute: int = 150) -> Optional[str]:
    """
    Estimate meeting duration based on word count.
    Average speaking pace is ~150 words/minute.

    Args:
        text: Transcript text.
        words_per_minute: Average speaking speed.

    Returns:
        Estimated duration string like "~30 minutes".
    """
    word_count = len(text.split())
    minutes = word_count / words_per_minute

    if minutes < 1:
        return "< 1 minute"
    elif minutes < 60:
        return f"~{int(minutes)} minutes"
    else:
        hours = int(minutes // 60)
        mins = int(minutes % 60)
        return f"~{hours}h {mins}m"
