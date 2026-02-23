# ──────────────────────────────────────────────
# Embedding Model Wrapper
# ──────────────────────────────────────────────
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from app.config import EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION

# Singleton: load model once and reuse
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Load the embedding model (cached after first call)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Generate embeddings for a list of texts.

    Args:
        texts: List of text strings to embed.

    Returns:
        numpy array of shape (len(texts), EMBEDDING_DIMENSION).
    """
    model = _get_model()
    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalization for cosine similarity
    )
    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """
    Generate embedding for a single query.

    Args:
        query: The query string.

    Returns:
        numpy array of shape (1, EMBEDDING_DIMENSION).
    """
    model = _get_model()
    embedding = model.encode(
        [query],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embedding.astype(np.float32)


def get_embedding_dimension() -> int:
    """Return the dimension of the embedding model."""
    return EMBEDDING_DIMENSION
