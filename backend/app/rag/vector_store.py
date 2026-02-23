# ──────────────────────────────────────────────
# FAISS Vector Store
# ──────────────────────────────────────────────
import os
import json
import faiss
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
from app.config import FAISS_INDEX_PATH, EMBEDDING_DIMENSION


class VectorStore:
    """
    FAISS-based vector store for document embeddings.

    Stores embeddings in a FAISS index and maintains a parallel
    metadata list to map vector IDs back to document chunks.
    """

    def __init__(self, index_path: str = FAISS_INDEX_PATH):
        self.index_path = index_path
        self.index_file = f"{index_path}.index"
        self.meta_file = f"{index_path}.meta.json"
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: List[Dict] = []
        self._load_or_create()

    def _load_or_create(self):
        """Load existing index from disk, or create a new one."""
        if os.path.exists(self.index_file) and os.path.exists(self.meta_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.meta_file, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            # Inner Product (IP) index — works with L2-normalized vectors
            # to give cosine similarity
            self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
            self.metadata = []

    def add(self, embeddings: np.ndarray, metadata_list: List[Dict]):
        """
        Add embeddings and their metadata to the store.

        Args:
            embeddings: numpy array of shape (n, dim).
            metadata_list: List of metadata dicts, one per embedding.
                           Each should contain: text, document_id, page_number, chunk_index.
        """
        if len(embeddings) != len(metadata_list):
            raise ValueError("Embeddings and metadata must have the same length.")

        self.index.add(embeddings)
        self.metadata.extend(metadata_list)
        self._save()

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Search for the most similar chunks.

        Args:
            query_embedding: numpy array of shape (1, dim).
            top_k: Number of results to return.

        Returns:
            List of dicts with 'score' and the original metadata fields.
        """
        if self.index.ntotal == 0:
            return []

        # Clamp top_k to available vectors
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            result = {**self.metadata[idx], "score": float(score)}
            results.append(result)

        return results

    def delete_by_document_id(self, document_id: str):
        """
        Remove all vectors associated with a document.

        Since FAISS IndexFlat doesn't support deletion, we rebuild
        the index without the deleted document's vectors.
        """
        # Find indices to keep
        keep_indices = [
            i for i, m in enumerate(self.metadata)
            if m.get("document_id") != document_id
        ]

        if len(keep_indices) == len(self.metadata):
            return  # Nothing to delete

        if len(keep_indices) == 0:
            # All vectors belong to this document — reset
            self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
            self.metadata = []
            self._save()
            return

        # Reconstruct vectors for items we're keeping
        all_vectors = np.array(
            [self.index.reconstruct(i) for i in keep_indices],
            dtype=np.float32,
        )
        new_metadata = [self.metadata[i] for i in keep_indices]

        # Rebuild index
        self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        self.index.add(all_vectors)
        self.metadata = new_metadata
        self._save()

    def _save(self):
        """Persist index and metadata to disk."""
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, self.index_file)
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False)

    @property
    def total_vectors(self) -> int:
        """Return the number of vectors in the index."""
        return self.index.ntotal if self.index else 0
