# ──────────────────────────────────────────────
# RAG Retriever — Orchestrates the full pipeline
# ──────────────────────────────────────────────
import uuid
from typing import List, Dict, Optional
from app.rag.pdf_processor import extract_text_from_pdf, get_pdf_metadata
from app.rag.chunker import chunk_pages
from app.rag.embeddings import embed_texts, embed_query
from app.rag.vector_store import VectorStore
from app.llm.ollama_client import get_ollama_client
from app.llm.prompts import RAG_PROMPT_TEMPLATE
from app.config import TOP_K_RESULTS

# Singleton vector store
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get or create the vector store singleton."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def ingest_document(file_path: str, document_id: str, filename: str) -> Dict:
    """
    Full ingestion pipeline: PDF → Chunks → Embeddings → FAISS.

    Args:
        file_path: Path to the uploaded PDF.
        document_id: Unique ID for the document.
        filename: Original filename.

    Returns:
        Dict with ingestion statistics.
    """
    # 1. Extract text from PDF
    pages = extract_text_from_pdf(file_path)
    if not pages:
        raise ValueError("No text could be extracted from the PDF.")

    # 2. Chunk the text
    chunks = chunk_pages(pages)
    if not chunks:
        raise ValueError("No chunks were produced from the document.")

    # 3. Get the text for embedding
    chunk_texts = [c["text"] for c in chunks]

    # 4. Generate embeddings
    embeddings = embed_texts(chunk_texts)

    # 5. Build metadata for each chunk
    metadata_list = [
        {
            "text": c["text"],
            "document_id": document_id,
            "filename": filename,
            "page_number": c["page_number"],
            "chunk_index": c["chunk_index"],
        }
        for c in chunks
    ]

    # 6. Add to vector store
    vs = get_vector_store()
    vs.add(embeddings, metadata_list)

    # 7. Get PDF metadata
    pdf_meta = get_pdf_metadata(file_path)

    return {
        "document_id": document_id,
        "filename": filename,
        "pages_extracted": len(pages),
        "chunks_created": len(chunks),
        "total_vectors": vs.total_vectors,
        "pdf_metadata": pdf_meta,
    }


def remove_document(document_id: str):
    """Remove a document's vectors from the store."""
    vs = get_vector_store()
    vs.delete_by_document_id(document_id)


async def query_documents(
    question: str,
    top_k: int = TOP_K_RESULTS,
) -> Dict:
    """
    Full query pipeline: Question → Embed → Search → LLM → Answer.

    Args:
        question: The user's question.
        top_k: Number of chunks to retrieve.

    Returns:
        Dict with 'answer' and 'sources'.
    """
    vs = get_vector_store()

    if vs.total_vectors == 0:
        return {
            "answer": "No documents have been uploaded yet. Please upload documents first.",
            "sources": [],
        }

    # 1. Embed the query
    q_embedding = embed_query(question)

    # 2. Search FAISS
    results = vs.search(q_embedding, top_k=top_k)

    if not results:
        return {
            "answer": "No relevant information found in the uploaded documents.",
            "sources": [],
        }

    # 3. Build context from retrieved chunks
    context_parts = []
    for i, r in enumerate(results, 1):
        context_parts.append(
            f"[Source {i} — {r['filename']}, Page {r['page_number']}]\n{r['text']}"
        )
    context = "\n\n".join(context_parts)

    # 4. Build the prompt
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

    # 5. Generate answer with LLM
    llm = get_ollama_client()
    answer = await llm.generate(prompt)

    # 6. Build source references
    sources = [
        {
            "filename": r["filename"],
            "page_number": r["page_number"],
            "chunk_index": r["chunk_index"],
            "score": r["score"],
            "snippet": r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"],
        }
        for r in results
    ]

    return {
        "answer": answer,
        "sources": sources,
    }
