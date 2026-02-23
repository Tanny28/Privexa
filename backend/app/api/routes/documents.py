# ──────────────────────────────────────────────
# Document Routes — Upload, List, Delete
# ──────────────────────────────────────────────
import os
import uuid
import shutil
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List
from app.db.database import get_db
from app.db.repositories import (
    create_document_record,
    list_documents,
    get_document_by_id,
    delete_document_record,
)
from app.api.dependencies import require_permission
from app.rag.retriever import ingest_document, remove_document, get_vector_store
from app.models.user import User
from app.config import UPLOAD_DIR, ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB

router = APIRouter(prefix="/documents", tags=["Documents"])


# ─── Schemas ─────────────────────────────────

class DocumentResponse(BaseModel):
    id: str
    filename: str
    page_count: int
    chunk_count: int
    uploaded_by: str
    uploaded_at: str | None = None


class UploadResponse(BaseModel):
    message: str
    document: DocumentResponse
    ingestion_stats: dict


class StatsResponse(BaseModel):
    total_documents: int
    total_vectors: int


# ─── Endpoints ───────────────────────────────

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(require_permission("upload_documents")),
    db: Session = Depends(get_db),
):
    """Upload a PDF document and ingest it into the RAG pipeline."""
    # Validate file extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only PDF files are allowed. Got: {ext}",
        )

    # Validate file size
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Max: {MAX_FILE_SIZE_MB}MB, Got: {size_mb:.1f}MB",
        )

    # Save file to disk
    document_id = str(uuid.uuid4())
    safe_filename = f"{document_id}{ext}"
    file_path = str(UPLOAD_DIR / safe_filename)

    with open(file_path, "wb") as f:
        f.write(contents)

    # Ingest into RAG pipeline
    try:
        stats = ingest_document(file_path, document_id, file.filename)
    except Exception as e:
        # Clean up the file if ingestion fails
        os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {str(e)}",
        )

    # Save metadata to database
    doc_record = create_document_record(
        db=db,
        document_id=document_id,
        filename=file.filename,
        file_path=file_path,
        page_count=stats["pages_extracted"],
        chunk_count=stats["chunks_created"],
        uploaded_by=current_user.id,
    )

    return UploadResponse(
        message="Document uploaded and indexed successfully.",
        document=DocumentResponse(
            id=doc_record.id,
            filename=doc_record.filename,
            page_count=doc_record.page_count,
            chunk_count=doc_record.chunk_count,
            uploaded_by=current_user.username,
        ),
        ingestion_stats=stats,
    )


@router.get("/", response_model=List[DocumentResponse])
def get_documents(
    current_user: User = Depends(require_permission("list_documents")),
    db: Session = Depends(get_db),
):
    """List all uploaded documents."""
    docs = list_documents(db)
    return [
        DocumentResponse(
            id=d.id,
            filename=d.filename,
            page_count=d.page_count,
            chunk_count=d.chunk_count,
            uploaded_by=d.uploaded_by,
            uploaded_at=str(d.uploaded_at) if d.uploaded_at else None,
        )
        for d in docs
    ]


@router.delete("/{document_id}")
def delete_document(
    document_id: str,
    current_user: User = Depends(require_permission("delete_documents")),
    db: Session = Depends(get_db),
):
    """Delete a document and its vectors."""
    doc = get_document_by_id(db, document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    # Remove vectors from FAISS
    remove_document(document_id)

    # Remove file from disk
    if os.path.exists(doc.file_path):
        os.remove(doc.file_path)

    # Remove from database
    delete_document_record(db, document_id)

    return {"message": "Document deleted successfully.", "document_id": document_id}


@router.get("/stats", response_model=StatsResponse)
def get_stats(
    current_user: User = Depends(require_permission("list_documents")),
    db: Session = Depends(get_db),
):
    """Get document and vector store statistics."""
    docs = list_documents(db)
    vs = get_vector_store()
    return StatsResponse(
        total_documents=len(docs),
        total_vectors=vs.total_vectors,
    )
